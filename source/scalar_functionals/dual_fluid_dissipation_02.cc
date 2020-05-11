#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dual_fluid_dissipation_02.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualFluidDissipation02<spacedim>::DualFluidDissipation02(	const vector<DependentField<spacedim,spacedim>>	e_omega,
															const set<types::material_id>					domain_of_integration,
															const Quadrature<spacedim>						quadrature,
															GlobalDataIncrementalFE<spacedim>&				global_data,
															const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DualFluidDissipation02", 1, 3),
global_data(&global_data),
method(method)
{
}

template<unsigned int spacedim>
bool
DualFluidDissipation02<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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

	double c_F_approx, c_plus_approx, c_minus_approx;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		c_F_approx     = e_omega_ref_sets[0][13];
		c_plus_approx  = e_omega_ref_sets[0][14];
		c_minus_approx = e_omega_ref_sets[0][15];
	}
	//corrector step of method == 1
	else
	{
		c_F_approx     = hidden_vars[0];
		c_plus_approx  = hidden_vars[1];
		c_minus_approx = hidden_vars[2];
	}

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
		for(unsigned int i = 0; i < 3; ++i)
			hidden_vars[i] = (e_omega[i + 13] + e_omega_ref_sets[0][i + 13])*0.5;


	Tensor<1,3> eta_1, lambda;
	for(unsigned int m = 0; m < 3; ++m)
	{
		eta_1[m] = e_omega[m+3] + c_plus_approx/c_F_approx * e_omega[m+6] + c_minus_approx/c_F_approx * e_omega[m+9];
		lambda[m] = e_omega[m];
	}

	const double eta_F = e_omega[12];
	const double c_F = e_omega[13];
	const double c_F_ref = e_omega_ref_sets[0][13];

	if(get<0>(requested_quantities))
		h_omega = lambda * eta_1 - eta_F * (c_F - c_F_ref);

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			h_omega_1[m] =   eta_1[m];
			h_omega_1[m+3] = lambda[m];
			h_omega_1[m+6] = lambda[m] * c_plus_approx / c_F_approx;
			h_omega_1[m+9] = lambda[m] * c_minus_approx / c_F_approx;
		}
		h_omega_1[12] = -(c_F - c_F_ref);
		h_omega_1[13] = -eta_F;
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
		{
			h_omega_2(m + 3, m) = h_omega_2(m, m + 3)  = 1.0;
			h_omega_2(m + 6, m) = h_omega_2(m, m + 6)  = c_plus_approx / c_F_approx;
			h_omega_2(m + 9, m) = h_omega_2(m, m + 9)  = c_minus_approx / c_F_approx;
		}
		h_omega_2(12, 13) = -1.0;
		h_omega_2(13, 12) = -1.0;
	}

	return false;
}

template class DualFluidDissipation02<2>;
template class DualFluidDissipation02<3>;
