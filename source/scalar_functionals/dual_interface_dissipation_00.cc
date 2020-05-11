#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <incremental_fe/scalar_functionals/dual_interface_dissipation_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualInterfaceDissipation00<spacedim>::DualInterfaceDissipation00(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
																	const set<types::material_id>						domain_of_integration,
																	const Quadrature<spacedim-1>						quadrature,
																	GlobalDataIncrementalFE<spacedim>&					global_data,
																	const double										d)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "DualInterfaceDissipation00", 1),
global_data(&global_data),
d(d)
{
}

template<unsigned int spacedim>
bool
DualInterfaceDissipation00<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
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

	if(get<0>(requested_quantities))
	{
		h_sigma = -0.5 * d * delta_t * e_sigma[0] * e_sigma[0];
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(1);
		h_sigma_1[0] = -d*delta_t*e_sigma[0];
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(1,1);
		h_sigma_2(0,0) = -d*delta_t;
	}
	return false;
}

template class DualInterfaceDissipation00<2>;
template class DualInterfaceDissipation00<3>;
