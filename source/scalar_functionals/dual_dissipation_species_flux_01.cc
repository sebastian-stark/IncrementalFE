#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dual_dissipation_species_flux_01.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualDissipationSpeciesFlux01<spacedim>::DualDissipationSpeciesFlux01(	const vector<DependentField<spacedim,spacedim>>	e_omega,
																const set<types::material_id>					domain_of_integration,
																const Quadrature<spacedim>						quadrature,
																GlobalDataIncrementalFE<spacedim>&				global_data,
																const double									D,
																const unsigned int								method)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DualDissipationSpeciesFlux01", 1, 10),
global_data(&global_data),
D(D),
method(method)
{
}

template<unsigned int spacedim>
bool
DualDissipationSpeciesFlux01<spacedim>::get_h_omega(const Vector<double>&			e_omega,
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
	Tensor<1, 3> eta_1;
	for(unsigned int m = 0; m < 3; ++m)
		eta_1[m] = e_omega[m];
	const double c = e_omega[3];
	const double c_ref = e_omega_ref_sets[0][3];
	const double eta = e_omega[4];

	double c_approx;
	Tensor<2, 3> F, C, C_inv;
	//method == 0 or predictor step of method == 2
	if(method == 0 || global_data->get_predictor_step())
	{
		c_approx = e_omega_ref_sets[0][3];
		F[0][0] = e_omega_ref_sets[0][5];
		F[0][1] = e_omega_ref_sets[0][6];
		F[0][2] = e_omega_ref_sets[0][7];
		F[1][0] = e_omega_ref_sets[0][8];
		F[1][1] = e_omega_ref_sets[0][9];
		F[1][2] = e_omega_ref_sets[0][10];
		F[2][0] = e_omega_ref_sets[0][11];
		F[2][1] = e_omega_ref_sets[0][12];
		F[2][2] = e_omega_ref_sets[0][13];
	}
	//corrector step of method == 1
	else
	{
		c_approx = hidden_vars[0];
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

	//invert C
	FullMatrix<double> C_temp(3,3);
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int n = 0; n < 3; ++n)
			C_temp(m,n) = C[m][n];
	C_temp.invert(C_temp);
	for(unsigned int m = 0; m < 3; ++m)
		for(unsigned int n = 0; n < 3; ++n)
			C_inv[m][n] = C_temp(m,n);

	//make sure that predicted values are stored in hidden variables
	if(global_data->get_predictor_step())
	{
		hidden_vars[0] = (e_omega[3] + e_omega_ref_sets[0][3])*0.5;
		for(unsigned int i = 0; i < 9; ++i)
			hidden_vars[i + 1] = (e_omega[i + 5] + e_omega_ref_sets[0][i + 5])*0.5;
	}

	Tensor<1, 3> C_inv_eta_1 = C_inv * eta_1;
	const double eta_1_C_inv_eta_1 = C_inv_eta_1 * eta_1;

	if(get<0>(requested_quantities))
		h_omega = -0.5 * delta_t * D * c_approx * eta_1_C_inv_eta_1 - eta * (c-c_ref);

	if(get<1>(requested_quantities))
	{
		h_omega_1.reinit(this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
			h_omega_1[m] = -delta_t * D * c_approx * C_inv_eta_1[m];
		h_omega_1[3] = -eta;
		h_omega_1[4] = -(c-c_ref);
	}

	if(get<2>(requested_quantities))
	{
		h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				h_omega_2(m, n) = -delta_t * D * c_approx * C_inv[m][n];
		h_omega_2(3,4) = -1.0;
		h_omega_2(4,3) = -1.0;
	}

	return false;
}

template<unsigned int spacedim>
double
DualDissipationSpeciesFlux01<spacedim>::get_maximum_step(	const Vector<double>& 			e_omega,
															const vector<Vector<double>>&	/*e_omega_ref_sets*/,
															const Vector<double>& 			delta_e_omega,
															const Vector<double>& 			/*hidden_vars*/,
															const Point<spacedim>& 			/*x*/)

const
{
	double max_step = - e_omega[3] / delta_e_omega[3];
	if(isnan(max_step) || (max_step < 0.0))
		return DBL_MAX;
	else
		return max_step;
}

template class DualDissipationSpeciesFlux01<2>;
template class DualDissipationSpeciesFlux01<3>;
