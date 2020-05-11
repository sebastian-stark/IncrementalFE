#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <cfloat>
#include <incremental_fe/scalar_functionals/interface_dissipation_02.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

namespace
{
	void
	compute_delta_and_derivatives(const double& alpha, const double& i, double& val, double& d1, double& d2)
	{
		//solve for x
		double eta = 0.0;

		if(fabs(i) > DBL_MIN)
		{
			//initial guess
			if(i > 0)
				eta = max(0.0, log(i)/alpha);
			else
				eta = min(0.0, log(-i)/(alpha-1.0));

			//iterate
			double dx = 0.0;
			double r, t1, t2;
			double tol = fabs(i) > 1.0 ? 1e-12 * fabs(i) : 1e-12;
			unsigned int iter = 0;
			for(;;)
			{
				++iter;
				t1 = exp(alpha * eta);
				t2 = exp((alpha - 1.0) * eta);
				r = t1 - t2 - i;
				dx = -r/( alpha * t1 + (1.0 - alpha) * t2 );
				eta += dx;
				if(fabs(r) < tol)
					break;
				if(iter > 100)
					cout << "No convergence in iteration, exiting!" << endl;
			}
		}

		//return values and derivatives
		const double t1 = exp(alpha * eta);
		const double t2 = exp((alpha - 1.0) * eta);
		val = i * eta - ( 1.0/alpha * t1 + 1.0/(1.0 - alpha) * t2 ) + ( 1.0/alpha + 1.0/(1.0 - alpha) );
		d1 = eta;
		d2 = 1.0 / ( alpha * t1 + (1.0 - alpha) * t2);
	}
}

template<unsigned int spacedim>
InterfaceDissipation02<spacedim>::InterfaceDissipation02(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
															const set<types::material_id>						domain_of_integration,
															const Quadrature<spacedim-1>						quadrature,
															GlobalDataIncrementalFE<spacedim>&					global_data,
															const double										alpha,
															const double										RT,
															const double										i0,
															const unsigned int									method)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "InterfaceDissipation02", 1, 9),
global_data(&global_data),
alpha(alpha),
RT(RT),
i0(i0),
method(method)
{
}

template<unsigned int spacedim>
bool
InterfaceDissipation02<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
												const vector<Vector<double>>&	e_sigma_ref_sets,
												Vector<double>& 				hidden_vars,
												const Point<spacedim>& 			/*x*/,
												const Tensor<1,spacedim>& 		n,
												double& 						h_sigma,
												Vector<double>& 				h_sigma_1,
												FullMatrix<double>& 			h_sigma_2,
												const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));
	Assert(e_sigma_ref_sets.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref_sets[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	Tensor<2, 3> F, C, C_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		F[0][0] = e_sigma_ref_sets[0][3];
		F[0][1] = e_sigma_ref_sets[0][4];
		F[0][2] = e_sigma_ref_sets[0][5];
		F[1][0] = e_sigma_ref_sets[0][6];
		F[1][1] = e_sigma_ref_sets[0][7];
		F[1][2] = e_sigma_ref_sets[0][8];
		F[2][0] = e_sigma_ref_sets[0][9];
		F[2][1] = e_sigma_ref_sets[0][10];
		F[2][2] = e_sigma_ref_sets[0][11];
	}
	//corrector step of method == 1
	else
	{
		F[0][0] = hidden_vars[0];
		F[0][1] = hidden_vars[1];
		F[0][2] = hidden_vars[2];
		F[1][0] = hidden_vars[3];
		F[1][1] = hidden_vars[4];
		F[1][2] = hidden_vars[5];
		F[2][0] = hidden_vars[6];
		F[2][1] = hidden_vars[7];
		F[2][2] = hidden_vars[8];
	}
	C = transpose(F) * F;

	//invert C
	FullMatrix<double> C_temp(3,3);
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int n = 0; n < 3; ++n)
			C_temp(m,n) = C[m][n];
	C_temp.invert(C_temp);
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int n = 0; n < 3; ++n)
			C_inv[m][n] = C_temp(m,n);

	//compute determinant of F
	FullMatrix<double> F_temp(3,3);
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int n = 0; n < 3; ++n)
			F_temp(m,n) = F[m][n];
	const double J = F_temp.determinant();
	Assert(J>0, ExcMessage("Detected negative determinant of deformation gradient!"));

	//normal
	Tensor<1,3> n_;
	for(unsigned int m = 0; m < spacedim; ++m)
		n_[m] = n[m];

	const double prefactor = J * sqrt( (C_inv * n_) * n_ );

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 9; ++i)
			hidden_vars[i] = (e_sigma[i + 3] + e_sigma_ref_sets[0][i + 3])*0.5;


	const double delta_t = global_data->get_t() - global_data->get_t_ref();
	const double nx = n[0];
	const double ny = n[1];
	const double nz = spacedim==3 ? n[2] : 0.0;

	const double i_n = ( (e_sigma[0]-e_sigma_ref_sets[0][0])*nx + (e_sigma[1]-e_sigma_ref_sets[0][1])*ny + (e_sigma[2]-e_sigma_ref_sets[0][2])*nz ) / delta_t / prefactor;

	//compute values and derivatives
	double val, d1, d2;
	compute_delta_and_derivatives(alpha, i_n/i0, val, d1, d2);
	val *= RT * (delta_t * i0 * prefactor);
	d1  *= RT;
	d2  *= RT / (delta_t * i0 * prefactor);

	if(get<0>(requested_quantities))
	{
		h_sigma = val;
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(12);
		h_sigma_1[0] = d1*nx;
		h_sigma_1[1] = d1*ny;
		h_sigma_1[2] = d1*nz;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(12,12);
		h_sigma_2(0,0) = d2*nx*nx;
		h_sigma_2(1,1) = d2*ny*ny;
		h_sigma_2(2,2) = d2*nz*nz;
		h_sigma_2(0,1) = h_sigma_2(1,0) = d2*nx*ny;
		h_sigma_2(1,2) = h_sigma_2(2,1) = d2*ny*nz;
		h_sigma_2(0,2) = h_sigma_2(2,0) = d2*nx*nz;
	}

	return false;
}

template class InterfaceDissipation02<2>;
template class InterfaceDissipation02<3>;
