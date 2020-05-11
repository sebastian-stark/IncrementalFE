#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dual_fluid_dissipation_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

namespace
{
	double get_J(const Tensor<2, 3>& F)
	{
		return	  F[0][0] * F[1][1] * F[2][2]
				+ F[0][1] * F[1][2] * F[2][0]
				+ F[0][2] * F[1][0] * F[2][1]
				- F[0][2] * F[1][1] * F[2][0]
				- F[0][1] * F[1][0] * F[2][2]
				- F[0][0] * F[1][2] * F[2][1];
	}
}

template<unsigned int spacedim>
DualFluidDissipation00<spacedim>::DualFluidDissipation00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									D,
													const double									V_m_F,
													const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DualFluidDissipation00", 1, 12),
global_data(&global_data),
D(D),
V_m_F(V_m_F),
method(method)
{
}

template<unsigned int spacedim>
bool
DualFluidDissipation00<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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

	double delta_t = global_data->get_t()-global_data->get_t_ref();
	Assert(delta_t > 0, ExcMessage("A non-positive time increment has been detected!"));

	double c_F_approx, c_plus_approx, c_minus_approx;
	Tensor<2, 3> F_approx, C_approx, C_approx_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		c_F_approx     = e_omega_ref_sets[0][12];
		c_plus_approx  = e_omega_ref_sets[0][13];
		c_minus_approx = e_omega_ref_sets[0][14];
		F_approx[0][0] = e_omega_ref_sets[0][15];
		F_approx[0][1] = e_omega_ref_sets[0][16];
		F_approx[0][2] = e_omega_ref_sets[0][17];
		F_approx[1][0] = e_omega_ref_sets[0][18];
		F_approx[1][1] = e_omega_ref_sets[0][19];
		F_approx[1][2] = e_omega_ref_sets[0][20];
		F_approx[2][0] = e_omega_ref_sets[0][21];
		F_approx[2][1] = e_omega_ref_sets[0][22];
		F_approx[2][2] = e_omega_ref_sets[0][23];
	}
	//corrector step of method == 1
	else
	{
		c_F_approx     = hidden_vars[0];
		c_plus_approx  = hidden_vars[1];
		c_minus_approx = hidden_vars[2];
		F_approx[0][0] = hidden_vars[3];
		F_approx[0][1] = hidden_vars[4];
		F_approx[0][2] = hidden_vars[5];
		F_approx[1][0] = hidden_vars[6];
		F_approx[1][1] = hidden_vars[7];
		F_approx[1][2] = hidden_vars[8];
		F_approx[2][0] = hidden_vars[9];
		F_approx[2][1] = hidden_vars[10];
		F_approx[2][2] = hidden_vars[11];
	}
	C_approx = transpose(F_approx) * F_approx;
	C_approx_inv = invert(C_approx);

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 12; ++i)
			hidden_vars[i] = (e_omega[i + 12] + e_omega_ref_sets[0][i + 12])*0.5;

	const double n_F_approx = c_F_approx / get_J(F_approx) * V_m_F;


	Tensor<1,3> eta_1;
	for(unsigned int m = 0; m < 3; ++m)
		eta_1[m] = e_omega[m] + c_plus_approx/c_F_approx * e_omega[m+3] + c_minus_approx/c_F_approx * e_omega[m+6];
	const double eta_F = e_omega[9];
	const double c_F = e_omega[12];
	const double c_F_ref = e_omega_ref_sets[0][12];

	Tensor<1, 3> C_approx_inv_eta_1 = C_approx_inv * eta_1;
	double eta_1_C_inv_eta_1 = C_approx_inv_eta_1 * eta_1;
	if(get<0>(requested_quantities))
		h_omega = - delta_t * D * c_F_approx / (2.0*n_F_approx) * eta_1_C_inv_eta_1 - eta_F * (c_F - c_F_ref);

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			h_omega_1[m] =   - delta_t * D * c_F_approx / n_F_approx * C_approx_inv_eta_1[m];
			h_omega_1[m+3] = - delta_t * D * c_F_approx / n_F_approx * C_approx_inv_eta_1[m] * c_plus_approx / c_F_approx;
			h_omega_1[m+6] = - delta_t * D * c_F_approx / n_F_approx * C_approx_inv_eta_1[m] * c_minus_approx / c_F_approx;
		}
		h_omega_1[9] = -(c_F - c_F_ref);
		h_omega_1[12] = -eta_F;
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			for(unsigned int n = 0; n < 3; ++n)
			{
				h_omega_2(m, n)     = 						-delta_t * D * c_F_approx / n_F_approx * C_approx_inv[m][n];
				h_omega_2(m+3, n+3) = 						-delta_t * D * c_F_approx / n_F_approx * C_approx_inv[m][n] * (c_plus_approx / c_F_approx) * (c_plus_approx / c_F_approx);
				h_omega_2(m+6, n+6) = 						-delta_t * D * c_F_approx / n_F_approx * C_approx_inv[m][n] * (c_minus_approx / c_F_approx) * (c_minus_approx / c_F_approx);
				h_omega_2(  m, n+3) = h_omega_2(n+3,   m) = -delta_t * D * c_F_approx / n_F_approx * C_approx_inv[m][n] * c_plus_approx / c_F_approx;
				h_omega_2(  m, n+6) = h_omega_2(n+6,   m) = -delta_t * D * c_F_approx / n_F_approx * C_approx_inv[m][n] * c_minus_approx / c_F_approx;
				h_omega_2(m+3, n+6) = h_omega_2(n+6, m+3) = -delta_t * D * c_F_approx / n_F_approx * C_approx_inv[m][n] * (c_plus_approx / c_F_approx) * (c_minus_approx / c_F_approx);

			}
		}
		h_omega_2( 9, 12) = -1.0;
		h_omega_2(12,  9) = -1.0;
	}

	return false;
}

template class DualFluidDissipation00<2>;
template class DualFluidDissipation00<3>;
