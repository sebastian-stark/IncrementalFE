#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <cfloat>
#include <incremental_fe/scalar_functionals/dual_interface_dissipation_02.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualInterfaceDissipation02<spacedim>::DualInterfaceDissipation02(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
																	const set<types::material_id>						domain_of_integration,
																	const Quadrature<spacedim-1>						quadrature,
																	GlobalDataIncrementalFE<spacedim>&					global_data,
																	const double										alpha,
																	const double										RT,
																	const double										i0,
																	const unsigned int									method,
																	const double										threshold)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "DualInterfaceDissipation02", 1, 9),
global_data(&global_data),
alpha(alpha),
RT(RT),
i0(i0),
method(method),
threshold(threshold)
{
}

template<unsigned int spacedim>
bool
DualInterfaceDissipation02<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
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
		F[0][0] = e_sigma_ref_sets[0][1];
		F[0][1] = e_sigma_ref_sets[0][2];
		F[0][2] = e_sigma_ref_sets[0][3];
		F[1][0] = e_sigma_ref_sets[0][4];
		F[1][1] = e_sigma_ref_sets[0][5];
		F[1][2] = e_sigma_ref_sets[0][6];
		F[2][0] = e_sigma_ref_sets[0][7];
		F[2][1] = e_sigma_ref_sets[0][8];
		F[2][2] = e_sigma_ref_sets[0][9];
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

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 9; ++i)
			hidden_vars[i] = (e_sigma[i + 1] + e_sigma_ref_sets[0][i + 1])*0.5;

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
	Assert(J > 0, ExcMessage("Detected negative determinant of deformation gradient!"));

	//normal
	Tensor<1,3> n_;
	for(unsigned int m = 0; m < spacedim; ++m)
		n_[m] = n[m];

	const double delta_t = global_data->get_t() - global_data->get_t_ref();

	const double eta = e_sigma[0];

	double t1, t2;

	if(eta / RT > threshold)
	{
		t1 = exp(alpha * threshold);
		t2 = exp( (alpha-1.0) * threshold);
	}
	else if(eta / RT < -threshold)
	{
		t1 = exp(-alpha * threshold);
		t2 = exp( -(alpha-1.0) * threshold);
	}
	else
	{
		t1 = exp(alpha * eta / RT);
		t2 = exp( (alpha-1.0) * eta /RT);
	}

	const double prefactor = -delta_t * i0 * RT * J * sqrt( (C_inv * n_) * n_ );

	if(get<0>(requested_quantities))
	{
		if(eta / RT > threshold)
		{
			h_sigma = t1/alpha + t2/(1.0-alpha) + (t1 - t2) * (eta/RT - threshold) + 0.5 * (alpha * t1 + (1.0-alpha) * t2) * (eta/RT - threshold) * (eta/RT - threshold);
		}
		else if(eta / RT < -threshold)
		{
			h_sigma = t1/alpha + t2/(1.0-alpha) + (t1 - t2) * (eta/RT + threshold) + 0.5 * (alpha * t1 + (1.0-alpha) * t2) * (eta/RT + threshold) * (eta/RT + threshold);
		}
		else
		{
			h_sigma = t1/alpha + t2/(1.0-alpha);
		}
		h_sigma *= prefactor;
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(10);
		if(eta / RT > threshold)
		{
			h_sigma_1[0] = (t1 - t2) + (alpha * t1 + (1.0-alpha) * t2) * (eta/RT - threshold);
		}
		else if(eta / RT < -threshold)
		{
			h_sigma_1[0] = (t1 - t2) + (alpha * t1 + (1.0-alpha) * t2) * (eta/RT + threshold);
		}
		else
		{
			h_sigma_1[0] = (t1 - t2);
		}
		h_sigma_1[0] *= prefactor / RT;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(10,10);
		if(eta / RT > threshold)
		{
			h_sigma_2(0,0) = alpha * t1 + (1.0-alpha) * t2;
		}
		else if(eta / RT < -threshold)
		{
			h_sigma_2(0,0) = alpha * t1 + (1.0-alpha) * t2;
		}
		else
		{
			h_sigma_2(0,0) = alpha * t1 + (1.0 - alpha) * t2;
		}
		h_sigma_2(0,0) *= prefactor / RT / RT;
	}

	return false;
}

template class DualInterfaceDissipation02<2>;
template class DualInterfaceDissipation02<3>;
