#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dual_fluid_dissipation_01.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualFluidDissipation01<spacedim>::DualFluidDissipation01(	const vector<DependentField<spacedim,spacedim>>	e_omega,
															const set<types::material_id>					domain_of_integration,
															const Quadrature<spacedim>						quadrature,
															GlobalDataIncrementalFE<spacedim>&				global_data,
															const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DualFluidDissipation01", 1, 11),
global_data(&global_data),
method(method)
{
}

template<unsigned int spacedim>
bool
DualFluidDissipation01<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
												const vector<Vector<double>>&	e_omega_ref_sets,
												Vector<double>&					hidden_vars,
												const Point<spacedim>&			/*x*/,
												double&							h_omega,
												Vector<double>&					h_omega_1,
												FullMatrix<double>&				h_omega_2,
												const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_omega.size() == this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e vector!"));
	Assert(e_omega_ref_sets.size() >= this->n_ref_sets,ExcMessage("Called get_h_omega with not enough datasets for the reference values of the independent fields!"));
	Assert(e_omega_ref_sets[0].size() == this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e_omega_ref_sets vector!"));

	Tensor<1,3> u_S_approx;
	Tensor<2,3> F_approx_inv;
	double c_plus_approx, c_minus_approx;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		c_plus_approx  = e_omega_ref_sets[0][12];
		c_minus_approx = e_omega_ref_sets[0][13];
		F_approx_inv[0][0] = e_omega_ref_sets[0][14];
		F_approx_inv[0][1] = e_omega_ref_sets[0][15];
		F_approx_inv[0][2] = e_omega_ref_sets[0][16];
		F_approx_inv[1][0] = e_omega_ref_sets[0][17];
		F_approx_inv[1][1] = e_omega_ref_sets[0][18];
		F_approx_inv[1][2] = e_omega_ref_sets[0][19];
		F_approx_inv[2][0] = e_omega_ref_sets[0][20];
		F_approx_inv[2][1] = e_omega_ref_sets[0][21];
		F_approx_inv[2][2] = e_omega_ref_sets[0][22];
	}
	//corrector step of method == 1
	else
	{
		c_plus_approx  = hidden_vars[0];
		c_minus_approx = hidden_vars[1];
		F_approx_inv[0][0] = hidden_vars[2];
		F_approx_inv[0][1] = hidden_vars[3];
		F_approx_inv[0][2] = hidden_vars[4];
		F_approx_inv[1][0] = hidden_vars[5];
		F_approx_inv[1][1] = hidden_vars[6];
		F_approx_inv[1][2] = hidden_vars[7];
		F_approx_inv[2][0] = hidden_vars[8];
		F_approx_inv[2][1] = hidden_vars[9];
		F_approx_inv[2][2] = hidden_vars[10];
	}
	F_approx_inv = invert(F_approx_inv);

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 11; ++i)
			hidden_vars[i] = (e_omega[i + 12] + e_omega_ref_sets[0][i + 12]) * 0.5;

	Tensor<1,3> eta_plus_1, eta_minus_1, delta_u_F, delta_u_S, u_F_ref, u_S_ref;
	for(unsigned int m = 0; m < 3; ++m)
	{
		eta_plus_1[m] = e_omega[m];
		eta_minus_1[m] = e_omega[m+3];
		delta_u_F[m] = e_omega[m+6] - e_omega_ref_sets[0][m+6];
		delta_u_S[m] = e_omega[m+9] - e_omega_ref_sets[0][m+9];
	}

	const Tensor<1,3> F_inv_u_F_u_S = F_approx_inv * (delta_u_F - delta_u_S);
	const Tensor<1,3> c_plus_eta_c_minus_eta_F_inv = (c_plus_approx * eta_plus_1 + c_minus_approx * eta_minus_1) * F_approx_inv;
	if(get<0>(requested_quantities))
		h_omega = 0.0;

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		for(unsigned int k = 0; k<3; ++k)
		{
			h_omega_1[k] = F_inv_u_F_u_S[k] * c_plus_approx;
			h_omega_1[k+3] = F_inv_u_F_u_S[k] * c_minus_approx;
			h_omega_1[k+6] = c_plus_eta_c_minus_eta_F_inv[k];
		}
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		for(unsigned int k = 0; k<3; ++k)
		{
			for(unsigned int l = 0; l < 3; ++l)
			{
				h_omega_2(k, l+6) = h_omega_2(l+6, k) = c_plus_approx * F_approx_inv[k][l];
				h_omega_2(k, l+9) = -c_plus_approx * F_approx_inv[k][l];
				h_omega_2(k+3, l+6) = h_omega_2(l+6, k+3) = c_minus_approx * F_approx_inv[k][l];
				h_omega_2(k+3, l+9) = -c_minus_approx * F_approx_inv[k][l];
			}
		}
	}

	return false;
}

template class DualFluidDissipation01<2>;
template class DualFluidDissipation01<3>;
