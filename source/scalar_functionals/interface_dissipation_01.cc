#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <cfloat>
#include <incremental_fe/scalar_functionals/interface_dissipation_01.h>

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
InterfaceDissipation01<spacedim>::InterfaceDissipation01(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
															const set<types::material_id>						domain_of_integration,
															const Quadrature<spacedim-1>						quadrature,
															GlobalDataIncrementalFE<spacedim>&					global_data,
															const double										alpha,
															const double										RT,
															const double										i0)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "InterfaceDissipation01", 1),
global_data(&global_data),
alpha(alpha),
RT(RT),
i0(i0)
{
}

template<unsigned int spacedim>
bool
InterfaceDissipation01<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
												const vector<Vector<double>>&	e_sigma_ref,
												Vector<double>& 				/*hidden_vars*/,
												const Point<spacedim>& 			/*x*/,
												const Tensor<1,spacedim>& 		n,
												double& 						h_sigma,
												Vector<double>& 				h_sigma_1,
												FullMatrix<double>& 			h_sigma_2,
												const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));
	Assert(e_sigma_ref.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	const double delta_t = global_data->get_t() - global_data->get_t_ref();
	const double nx = n[0];
	const double ny = n[1];
	const double nz = spacedim==3 ? n[2] : 0.0;

	const double i_n = ( (e_sigma[0]-e_sigma_ref[0][0])*nx + (e_sigma[1]-e_sigma_ref[0][1])*ny + (e_sigma[2]-e_sigma_ref[0][2])*nz ) / delta_t;

	//compute values and derivatives
	double val, d1, d2;
	compute_delta_and_derivatives(alpha, i_n/i0, val, d1, d2);
	val *= RT * (delta_t * i0);
	d1  *= RT;
	d2  *= RT / (delta_t * i0);

	if(get<0>(requested_quantities))
	{
		h_sigma = val;
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(3);
		h_sigma_1[0] = d1*nx;
		h_sigma_1[1] = d1*ny;
		h_sigma_1[2] = d1*nz;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(3,3);
		h_sigma_2(0,0) = d2*nx*nx;
		h_sigma_2(1,1) = d2*ny*ny;
		h_sigma_2(2,2) = d2*nz*nz;
		h_sigma_2(0,1) = h_sigma_2(1,0) = d2*nx*ny;
		h_sigma_2(1,2) = h_sigma_2(2,1) = d2*ny*nz;
		h_sigma_2(0,2) = h_sigma_2(2,0) = d2*nx*nz;
	}
	return false;
}

template class InterfaceDissipation01<2>;
template class InterfaceDissipation01<3>;
