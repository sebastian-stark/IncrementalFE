#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <cfloat>
#include <incremental_fe/scalar_functionals/solution_gel_interaction_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
SolutionGelInteraction00<spacedim>::SolutionGelInteraction00(	const vector<DependentField<spacedim-1,spacedim>>	e_sigma,
																const set<types::material_id>						domain_of_integration,
																const Quadrature<spacedim-1>						quadrature,
																GlobalDataIncrementalFE<spacedim>&					global_data,
																const double										V_m_F,
																const unsigned int									method)
:
ScalarFunctional<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, "SolutionGelInteraction00", 1, 9),
global_data(&global_data),
V_m_F(V_m_F),
method(method)
{
}

template<unsigned int spacedim>
bool
SolutionGelInteraction00<spacedim>::get_h_sigma(	const Vector<double>& 			e_sigma,
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

	Tensor<2,3> F_approx_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		F_approx_inv[0][0] = e_sigma_ref_sets[0][11];
		F_approx_inv[0][1] = e_sigma_ref_sets[0][12];
		F_approx_inv[0][2] = e_sigma_ref_sets[0][13];
		F_approx_inv[1][0] = e_sigma_ref_sets[0][14];
		F_approx_inv[1][1] = e_sigma_ref_sets[0][15];
		F_approx_inv[1][2] = e_sigma_ref_sets[0][16];
		F_approx_inv[2][0] = e_sigma_ref_sets[0][17];
		F_approx_inv[2][1] = e_sigma_ref_sets[0][18];
		F_approx_inv[2][2] = e_sigma_ref_sets[0][19];
	}
	//corrector step of method == 1
	else
	{
		F_approx_inv[0][0] = hidden_vars[0];
		F_approx_inv[0][1] = hidden_vars[1];
		F_approx_inv[0][2] = hidden_vars[2];
		F_approx_inv[1][0] = hidden_vars[3];
		F_approx_inv[1][1] = hidden_vars[4];
		F_approx_inv[1][2] = hidden_vars[5];
		F_approx_inv[2][0] = hidden_vars[6];
		F_approx_inv[2][1] = hidden_vars[7];
		F_approx_inv[2][2] = hidden_vars[8];
	}
	const double J = determinant(F_approx_inv);
	F_approx_inv = invert(F_approx_inv);

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 9; ++i)
			hidden_vars[i] = (e_sigma[i + 11] + e_sigma_ref_sets[0][i + 11]) * 0.5;

	//normal
	Tensor<1,3> n_;
	for(unsigned int m = 0; m < spacedim; ++m)
		n_[m] = n[m];
	const Tensor<1,3> s = n_ * F_approx_inv;
	const Tensor<1,3> tau = (V_m_F / J / (s*s) ) * s;

	Tensor<1,3> t, delta_u_S_g, delta_u_F_s;
	for(unsigned int m = 0; m < 3; ++m)
	{
		t[m] = e_sigma[m];
		delta_u_S_g[m] = e_sigma[m+3]-e_sigma_ref_sets[0][m+3];
		delta_u_F_s[m] = e_sigma[m+8]-e_sigma_ref_sets[0][m+8];
	}
	const double delta_I_F_g_n = e_sigma[6]-e_sigma_ref_sets[0][6];
	const double eta_F_g = e_sigma[7];

	if(get<0>(requested_quantities))
	{
		h_sigma = t * (delta_u_F_s - delta_u_S_g - delta_I_F_g_n * tau ) - eta_F_g * delta_I_F_g_n;
	}

	if(get<1>(requested_quantities))
	{
		h_sigma_1.reinit(e_sigma.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			h_sigma_1[m] = delta_u_F_s[m] - delta_u_S_g[m] - delta_I_F_g_n * tau[m] ;
			h_sigma_1[m+3] = -t[m];
			h_sigma_1[m+8] = t[m];
		}
		h_sigma_1[6] = - (t * tau) - eta_F_g;
		h_sigma_1[7] = - delta_I_F_g_n;
	}
	if(get<2>(requested_quantities))
	{
		h_sigma_2.reinit(e_sigma.size(),e_sigma.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			h_sigma_2(m, m+3) = h_sigma_2(m+3, m) = -1.0;
			h_sigma_2(m, m+8) = h_sigma_2(m+8, m) = 1.0;
			h_sigma_2(m, 6) = h_sigma_2(6, m) = -tau[m];
		}
		h_sigma_2(6, 7) = h_sigma_2(7, 6) = -1.0;
	}

	return false;
}

template class SolutionGelInteraction00<2>;
template class SolutionGelInteraction00<3>;
