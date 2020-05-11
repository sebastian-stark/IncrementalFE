#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <incremental_fe/scalar_functionals/zero_normal_flux_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
ZeroNormalFlux00<spacedim>::ZeroNormalFlux00(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
															const set<types::material_id>						domain_of_integration,
															const Quadrature<spacedim-1>						quadrature,
															GlobalDataIncrementalFE<spacedim>&					global_data)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "ZeroNormalFlux00", 0),
global_data(&global_data)
{
}

template<unsigned int spacedim>
bool
ZeroNormalFlux00<spacedim>::get_h_sigma(const Vector<double>& 			e_sigma,
										const vector<Vector<double>>&	/*e_sigma_ref*/,
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

	const double n_x = n[0];
	const double n_y = n[1];
	const double n_z = spacedim==3 ? n[2] : 0.0;
	const double lambda = e_sigma[0];
	const double I_x_dot = e_sigma[1];
	const double I_y_dot = e_sigma[2];
	const double I_z_dot = e_sigma[3];

	if(get<0>(requested_quantities))
	{
		h_sigma = lambda * (I_x_dot*n_x + I_y_dot*n_y + I_z_dot*n_z);
	}
	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(4);
		h_sigma_1[0] = I_x_dot*n_x + I_y_dot*n_y + I_z_dot*n_z;
		h_sigma_1[1] = lambda*n_x;
		h_sigma_1[2] = lambda*n_y;
		h_sigma_1[3] = lambda*n_z;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(4,4);
		h_sigma_2(0,1) = h_sigma_2(1,0) = n_x;
		h_sigma_2(0,2) = h_sigma_2(2,0) = n_y;
		h_sigma_2(0,3) = h_sigma_2(3,0) = n_z;
	}
	return false;
}

template class ZeroNormalFlux00<2>;
template class ZeroNormalFlux00<3>;
