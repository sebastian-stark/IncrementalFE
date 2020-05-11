#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/normal_flux_work_01.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
NormalFluxWork01<spacedim>::NormalFluxWork01(	const vector<DependentField<spacedim-1,spacedim>> e_sigma,
														const set<types::material_id> domain_of_integration,
														const Quadrature<spacedim-1> quadrature,
														GlobalDataIncrementalFE<spacedim>& global_data,
														Function<spacedim>&  functionPhi,
														const double alpha):
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "NormalFluxWork01", 1),
global_data(&global_data),
function_phi(functionPhi),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
NormalFluxWork01<spacedim>::get_h_sigma(const Vector<double>& 		e_sigma,
										const vector<Vector<double>>&	/*e_sigma_ref*/,
										Vector<double>& 				/*hidden_vars*/,
										const Point<spacedim>& 			x,
										const Tensor<1,spacedim>& 		/*n*/,
										double& 						h_sigma,
										Vector<double>& 				h_sigma_1,
										FullMatrix<double>& 			h_sigma_2,
										const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_sigma.size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e vector!"));

	const double time_old = function_phi.get_time();
	function_phi.set_time(global_data->get_t_ref());
	const double phi_0 = function_phi.value(x);
	function_phi.set_time(global_data->get_t());
	const double phi_1 = function_phi.value(x);
	function_phi.set_time(time_old);

	const double phi_tilde_alpha = alpha*phi_1 + (1.-alpha)*phi_0;

	const double i_n = e_sigma[0];

	if(get<0>(requested_quantities))
	{
		h_sigma = phi_tilde_alpha * i_n;
	}

	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(1);
		h_sigma_1[0] = phi_tilde_alpha;
	}

	if(get<2>(requested_quantities))
		h_sigma_2.reinit(1,1);

	return false;
}

template class NormalFluxWork01<2>;
template class NormalFluxWork01<3>;
