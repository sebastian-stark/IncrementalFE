#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dual_ion_dissipation_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualIonDissipation00<spacedim>::DualIonDissipation00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									D,
													const double									V_m_F,
													const unsigned int								method,
													const bool										no_solid_skeleton)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DualIonDissipation00", 1, 11),
global_data(&global_data),
D(D),
V_m_F(V_m_F),
method(method),
no_solid_skeleton(no_solid_skeleton)
{
}

template<unsigned int spacedim>
bool
DualIonDissipation00<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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

	double c_approx, c_F_approx;
	Tensor<2, 3> F_approx, C_approx, C_approx_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		c_approx       = e_omega_ref_sets[0][4];
		c_F_approx     = e_omega_ref_sets[0][5];
		F_approx[0][0] = e_omega_ref_sets[0][6];
		F_approx[0][1] = e_omega_ref_sets[0][7];
		F_approx[0][2] = e_omega_ref_sets[0][8];
		F_approx[1][0] = e_omega_ref_sets[0][9];
		F_approx[1][1] = e_omega_ref_sets[0][10];
		F_approx[1][2] = e_omega_ref_sets[0][11];
		F_approx[2][0] = e_omega_ref_sets[0][12];
		F_approx[2][1] = e_omega_ref_sets[0][13];
		F_approx[2][2] = e_omega_ref_sets[0][14];
	}
	//corrector step of method == 1
	else
	{
		c_approx       = hidden_vars[0];
		c_F_approx     = hidden_vars[1];
		F_approx[0][0] = hidden_vars[2];
		F_approx[0][1] = hidden_vars[3];
		F_approx[0][2] = hidden_vars[4];
		F_approx[1][0] = hidden_vars[5];
		F_approx[1][1] = hidden_vars[6];
		F_approx[1][2] = hidden_vars[7];
		F_approx[2][0] = hidden_vars[8];
		F_approx[2][1] = hidden_vars[9];
		F_approx[2][2] = hidden_vars[10];
	}
	C_approx = transpose(F_approx) * F_approx;
	C_approx_inv = invert(C_approx);

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 11; ++i)
			hidden_vars[i] = (e_omega[i + 4] + e_omega_ref_sets[0][i + 4])*0.5;

	//if there is no solid skeleton, ignore the previously determined value for c_F_approx and compute it from the incompressibility constraint

	const double n_F_approx = no_solid_skeleton ? 1.0 : c_F_approx / determinant(F_approx) * V_m_F;


	Tensor<1,3> eta_1;
	for(unsigned int m = 0; m < 3; ++m)
		eta_1[m] = e_omega[m];
	const double eta   = e_omega[3];
	const double c     = e_omega[4];
	const double c_ref = e_omega_ref_sets[0][4];

	Tensor<1, 3> C_approx_inv_eta_1 = C_approx_inv * eta_1;
	double eta_1_C_inv_eta_1 = C_approx_inv_eta_1 * eta_1;
	if(get<0>(requested_quantities))
		h_omega = - delta_t * D * c_approx / (2.0*n_F_approx) * eta_1_C_inv_eta_1 - eta * (c - c_ref);

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
			h_omega_1[m] =   - delta_t * D * c_approx / n_F_approx * C_approx_inv_eta_1[m];
		h_omega_1[3] = -(c - c_ref);
		h_omega_1[4] = -eta;
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			for(unsigned int n = 0; n < 3; ++n)
				h_omega_2(m, n) =- delta_t * D * c_approx / n_F_approx * C_approx_inv[m][n];
		}
		h_omega_2(3, 4) = -1.0;
		h_omega_2(4, 3) = -1.0;
	}

	return false;
}

template class DualIonDissipation00<2>;
template class DualIonDissipation00<3>;
