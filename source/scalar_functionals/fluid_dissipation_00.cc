#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/fluid_dissipation_00.h>

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
FluidDissipation00<spacedim>::FluidDissipation00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									D,
													const double									V_m_F,
													const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "FluidDissipation00", 1, 10),
global_data(&global_data),
D(D),
V_m_F(V_m_F),
method(method)
{
}

template<unsigned int spacedim>
bool
FluidDissipation00<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
											const vector<Vector<double>>&	e_omega_ref_sets,
											Vector<double>&					hidden_vars,
											const Point<spacedim>&			x,
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

	//no negative concentrations allowed -> if negative concentration return with error
	if(e_omega[3] < 0.0)
	{
		global_data->write_error_message("Negative species concentration in " + this->name, x);
		return true;
	}

	//average flux during time step
	Tensor<1, 3> i;
	for(unsigned int m = 0; m < 3; ++m)
		i[m] = (e_omega[m] - e_omega_ref_sets[0][m])/delta_t;

	double c;
	Tensor<2, 3> F, C;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		c = e_omega_ref_sets[0][3];
		F[0][0] = e_omega_ref_sets[0][4];
		F[0][1] = e_omega_ref_sets[0][5];
		F[0][2] = e_omega_ref_sets[0][6];
		F[1][0] = e_omega_ref_sets[0][7];
		F[1][1] = e_omega_ref_sets[0][8];
		F[1][2] = e_omega_ref_sets[0][9];
		F[2][0] = e_omega_ref_sets[0][10];
		F[2][1] = e_omega_ref_sets[0][11];
		F[2][2] = e_omega_ref_sets[0][12];
	}
	//corrector step of method == 1
	else
	{
		c = hidden_vars[0];
		F[0][0] = hidden_vars[1];
		F[0][1] = hidden_vars[2];
		F[0][2] = hidden_vars[3];
		F[1][0] = hidden_vars[4];
		F[1][1] = hidden_vars[5];
		F[1][2] = hidden_vars[6];
		F[2][0] = hidden_vars[7];
		F[2][1] = hidden_vars[8];
		F[2][2] = hidden_vars[9];
	}
	C = transpose(F) * F;

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 10; ++i)
			hidden_vars[i] = (e_omega[i + 3] + e_omega_ref_sets[0][i + 3])*0.5;

	const double n_F = c / get_J(F) * V_m_F;

	Tensor<1, 3> C_i = C * i;
	double i_C_i = C_i * i;
	if(get<0>(requested_quantities))
		h_omega = n_F * delta_t/(2.0*D*c) * i_C_i;

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
			h_omega_1[m] = n_F * 1.0/(D*c) * C_i[m];
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				h_omega_2(m, n) = n_F * 1.0/(delta_t*D*c) * C[m][n];
	}

	return false;
}

template class FluidDissipation00<2>;
template class FluidDissipation00<3>;
