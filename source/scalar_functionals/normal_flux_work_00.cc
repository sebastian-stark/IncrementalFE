#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/normal_flux_work_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;



template<unsigned int spacedim>
NormalFluxWork00<spacedim>::NormalFluxWork00(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
												const set<types::material_id>						domain_of_integration,
												const Quadrature<spacedim-1>						quadrature,
												GlobalDataIncrementalFE<spacedim>&					global_data,
												Function<spacedim>& 								function_phi,
												const double										alpha)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "NormalFluxWork00", 1),
global_data(&global_data),
function_phi(function_phi),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
NormalFluxWork00<spacedim>::get_h_sigma(const Vector<double>& 			e_sigma,
										const vector<Vector<double>>&	/*e_sigma_ref*/,
										Vector<double>& 				/*hidden_vars*/,
										const Point<spacedim>& 			x,
										const Tensor<1,spacedim>& 		n,
										double& 						h_sigma,
										Vector<double>& 				h_sigma_1,
										FullMatrix<double>& 			h_sigma_2,
										const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));

	const double time_old = function_phi.get_time();
	function_phi.set_time(global_data->get_t_ref());
	const double phi_ref = function_phi.value(x);
	function_phi.set_time(global_data->get_t());
	const double phi = function_phi.value(x);
	function_phi.set_time(time_old);

	const double phi_alpha = alpha*phi + (1.-alpha)*phi_ref;

	const double I_x = e_sigma[0];
	const double I_y = e_sigma[1];
	const double I_z = e_sigma[2];

	if(get<0>(requested_quantities))
	{
		if(spacedim == 3)
			h_sigma = phi_alpha * (I_x*n[0] + I_y*n[1] + I_z*n[2]);
		else
			h_sigma = phi_alpha * (I_x*n[0] + I_y*n[1]);
	}

	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(3);
		h_sigma_1[0] = phi_alpha*n[0];
		h_sigma_1[1] = phi_alpha*n[1];
		if(spacedim == 3)
			h_sigma_1[2] = phi_alpha*n[2];
		else
			h_sigma_1[2] = 0.0;
	}

		if(get<2>(requested_quantities))
			h_sigma_2.reinit(3,3);

	return false;
}

template class NormalFluxWork00<2>;
template class NormalFluxWork00<3>;
