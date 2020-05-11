#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/electrical_work_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
ElectricalWork00<spacedim>::ElectricalWork00(	const vector<DependentField<spacedim-1,spacedim>> e_sigma,
											const set<types::material_id> domain_of_integration,
											const Quadrature<spacedim-1> quadrature,
											GlobalDataIncrementalFE<spacedim>& global_data,
											double (*const functionPhi)(double, Point<spacedim>),
											const double alpha,
											const unsigned int method):
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "ElectricalWork00", 1),
global_data(&global_data),
function_phi(functionPhi),
alpha(alpha),
method(method)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
ElectricalWork00<spacedim>::get_h_sigma(const Vector<double>& 			e_sigma,
										const vector<Vector<double>>&	e_sigma_ref,
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
	Assert(e_sigma_ref.size() >= this->n_ref_sets,ExcMessage("Called get_h_sigma with not enough datasets for the reference values of the independent fields!"));
	Assert(e_sigma_ref[0].size() == this->e_sigma.size(),ExcMessage("Called get_h_sigma with invalid size of e_ref vector!"));

	if(method == 0)
	{
		const double t_alpha = alpha*global_data->get_t() + (1.-alpha)*global_data->get_t_ref();
		const double phi_tilde_alpha = function_phi(t_alpha, x);

		const double jump_Dx = e_sigma[0]*alpha + e_sigma_ref[0][0]*(1.-alpha);
		const double jump_Dy = e_sigma[1]*alpha + e_sigma_ref[0][1]*(1.-alpha);
		const double jump_Dz = e_sigma[2]*alpha + e_sigma_ref[0][2]*(1.-alpha);

		if(get<0>(requested_quantities))
		{
			if(spacedim == 3)
				h_sigma = phi_tilde_alpha * (jump_Dx*n[0] + jump_Dy*n[1] + jump_Dz*n[2]) / alpha;
			else
				h_sigma = phi_tilde_alpha * (jump_Dx*n[0] + jump_Dy*n[1]) / alpha;
		}

		if(get<1>(requested_quantities))
		{
			h_sigma_1.reinit(3);
			h_sigma_1[0] = phi_tilde_alpha*n[0];
			h_sigma_1[1] = phi_tilde_alpha*n[1];
			if(spacedim == 3)
				h_sigma_1[2] = phi_tilde_alpha*n[2];
			else
				h_sigma_1[2] = 0.0;
		}
		if(get<2>(requested_quantities))
			h_sigma_2.reinit(3,3);

	}
	else
	{
		const double phi_tilde_alpha = alpha*function_phi(global_data->get_t(), x) + (1.-alpha)*function_phi(global_data->get_t_ref(), x);

		const double jump_Dx = e_sigma[0];
		const double jump_Dy = e_sigma[1];
		const double jump_Dz = e_sigma[2];

		if(get<0>(requested_quantities))
		{
			if(spacedim == 3)
				h_sigma = phi_tilde_alpha * (jump_Dx*n[0] + jump_Dy*n[1] + jump_Dz*n[2]);
			else
				h_sigma = phi_tilde_alpha * (jump_Dx*n[0] + jump_Dy*n[1]);
		}

		if(get<1>(requested_quantities))
		{
			h_sigma_1.reinit(3);
			h_sigma_1[0] = phi_tilde_alpha*n[0];
			h_sigma_1[1] = phi_tilde_alpha*n[1];
			if(spacedim == 3)
				h_sigma_1[2] = phi_tilde_alpha*n[2];
			else
				h_sigma_1[2] = 0.0;
		}

		if(get<2>(requested_quantities))
			h_sigma_2.reinit(3,3);
	}

	return false;
}

template class ElectricalWork00<2>;
template class ElectricalWork00<3>;
