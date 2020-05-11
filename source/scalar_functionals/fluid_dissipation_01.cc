#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor.h>

#include <incremental_fe/scalar_functionals/fluid_dissipation_01.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
FluidDissipation01<spacedim>::FluidDissipation01(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									eta,
													const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "FluidDissipation01", 1, 9),
global_data(&global_data),
eta(eta),
method(method)
{
}

template<unsigned int spacedim>
bool
FluidDissipation01<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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

	//average velocity gradient during time step
	Tensor<2, 3> L;
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int n = 0; n < 3; ++n)
			L[m][n] = (e_omega[3*m + n] - e_omega_ref_sets[0][3*m + n])/delta_t;

	Tensor<2, 3> F_approx_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		F_approx_inv[0][0] = e_omega_ref_sets[0][9];
		F_approx_inv[0][1] = e_omega_ref_sets[0][10];
		F_approx_inv[0][2] = e_omega_ref_sets[0][11];
		F_approx_inv[1][0] = e_omega_ref_sets[0][12];
		F_approx_inv[1][1] = e_omega_ref_sets[0][13];
		F_approx_inv[1][2] = e_omega_ref_sets[0][14];
		F_approx_inv[2][0] = e_omega_ref_sets[0][15];
		F_approx_inv[2][1] = e_omega_ref_sets[0][16];
		F_approx_inv[2][2] = e_omega_ref_sets[0][17];
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
			hidden_vars[i] = (e_omega[i + 9] + e_omega_ref_sets[0][i + 9])*0.5;

	const Tensor<2,3> d = symmetrize(L * F_approx_inv);

	if(get<0>(requested_quantities))
		h_omega = J_approx * delta_t * eta / 2.0 * double_contract<0,0,1,1>(d, d);

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		const Tensor<2,3> F_inv_d = F_approx_inv * d;

		for(unsigned int n = 0; n < 3; ++n)
			for(unsigned int N = 0; N < 3; ++N)
				h_omega_1[n*3 + N] = J_approx * eta * F_inv_d[N][n];
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		const Tensor<2,3> C_approx_inv = F_approx_inv * transpose(F_approx_inv);
		for(unsigned int n = 0; n < 3; ++n)
		{
			for(unsigned int N = 0; N < 3; ++N)
				for(unsigned int k = 0; k < 3; ++k)
					for(unsigned int K = 0; K < 3; ++K)
					{
						h_omega_2(n*3 + N, k*3 + K) += 0.5 * J_approx * eta / delta_t * F_approx_inv[N][k] * F_approx_inv[K][n];
						if(n == k)
							h_omega_2(n*3 + N, k*3 + K) += 0.5 * J_approx * eta / delta_t * C_approx_inv[N][K];
					}
		}
	}

	return false;
}

template class FluidDissipation01<2>;
template class FluidDissipation01<3>;
