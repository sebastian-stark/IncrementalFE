#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <cfloat>
#include <incremental_fe/scalar_functionals/dual_interface_dissipation_01.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualInterfaceDissipation01<spacedim>::DualInterfaceDissipation01(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
																	const set<types::material_id>						domain_of_integration,
																	const Quadrature<spacedim-1>						quadrature,
																	GlobalDataIncrementalFE<spacedim>&					global_data,
																	const double										alpha,
																	const double										RT,
																	const double										i0,
																	const double										threshold)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "DualInterfaceDissipation01", 1),
global_data(&global_data),
alpha(alpha),
RT(RT),
i0(i0),
threshold(threshold)
{
}

template<unsigned int spacedim>
bool
DualInterfaceDissipation01<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
													const vector<Vector<double>>&	/*e_sigma_ref*/,
													Vector<double>& 				/*hidden_vars*/,
													const Point<spacedim>& 			/*x*/,
													const Tensor<1,spacedim>& 		/*n*/,
													double& 						h_sigma,
													Vector<double>& 				h_sigma_1,
													FullMatrix<double>& 			h_sigma_2,
													const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));

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
		t2 = exp( (alpha-1.0) * eta / RT);
	}

	const double prefactor = -delta_t * i0 * RT;


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
		h_sigma_1.reinit(1);
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
		h_sigma_2.reinit(1,1);
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

template class DualInterfaceDissipation01<2>;
template class DualInterfaceDissipation01<3>;
