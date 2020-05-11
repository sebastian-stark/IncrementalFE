#include <incremental_fe/scalar_functionals/incompressibility_constraint_01.h>

#include <math.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
IncompressibilityConstraint01<spacedim>::IncompressibilityConstraint01(	const vector<DependentField<spacedim,spacedim>>	e_omega,
																		const set<types::material_id>					domain_of_integration,
																		const Quadrature<spacedim>						quadrature,
																		GlobalDataIncrementalFE<spacedim>&				global_data,
																		const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "IncompressibilityConstraint01", 1, 9),
global_data(&global_data),
method(method)
{
}

template<unsigned int spacedim>
bool
IncompressibilityConstraint01<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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
	Assert(e_omega_ref_sets[0].size() == this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e_omega_ref vector!"));

	Tensor<2,3> F_approx_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		F_approx_inv[0][0] = e_omega_ref_sets[0][10];
		F_approx_inv[0][1] = e_omega_ref_sets[0][11];
		F_approx_inv[0][2] = e_omega_ref_sets[0][12];
		F_approx_inv[1][0] = e_omega_ref_sets[0][13];
		F_approx_inv[1][1] = e_omega_ref_sets[0][14];
		F_approx_inv[1][2] = e_omega_ref_sets[0][15];
		F_approx_inv[2][0] = e_omega_ref_sets[0][16];
		F_approx_inv[2][1] = e_omega_ref_sets[0][17];
		F_approx_inv[2][2] = e_omega_ref_sets[0][18];
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
	const double J_approx = determinant(F_approx_inv);
	F_approx_inv = invert(F_approx_inv);

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 9; ++i)
			hidden_vars[i] = (e_omega[i + 10] + e_omega_ref_sets[0][i + 10]) * 0.5;


	Tensor<2, 3> delta_L;
	delta_L[0][0] = e_omega[0] - e_omega_ref_sets[0][0];
	delta_L[0][1] = e_omega[1] - e_omega_ref_sets[0][1];
	delta_L[0][2] = e_omega[2] - e_omega_ref_sets[0][2];
	delta_L[1][0] = e_omega[3] - e_omega_ref_sets[0][3];
	delta_L[1][1] = e_omega[4] - e_omega_ref_sets[0][4];
	delta_L[1][2] = e_omega[5] - e_omega_ref_sets[0][5];
	delta_L[2][0] = e_omega[6] - e_omega_ref_sets[0][6];
	delta_L[2][1] = e_omega[7] - e_omega_ref_sets[0][7];
	delta_L[2][2] = e_omega[8] - e_omega_ref_sets[0][8];
	const double p = e_omega[9];

	const double delta_L_F_inv = double_contract<0,1,1,0>(delta_L, F_approx_inv);

	if(get<0>(requested_quantities))
	{
		h_omega = -p * delta_L_F_inv * J_approx;
	}

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				h_omega_1[m*3 + n] = -p * F_approx_inv[n][m] * J_approx;
		h_omega_1[9] = - delta_L_F_inv * J_approx ;
	}

	//second derivative
	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(e_omega.size(),e_omega.size());
		for(unsigned int k = 0; k < 3; ++k)
			for(unsigned int L = 0; L < 3; ++L)
				h_omega_2(k*3 + L, 9) = h_omega_2(9, k*3 + L) = - F_approx_inv[L][k] * J_approx;
	}

	return false;
}

template class IncompressibilityConstraint01<2>;
template class IncompressibilityConstraint01<3>;
