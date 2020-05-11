#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dissipation_species_flux_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DissipationSpeciesFlux00<spacedim>::DissipationSpeciesFlux00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
																const set<types::material_id>					domain_of_integration,
																const Quadrature<spacedim>						quadrature,
																GlobalDataIncrementalFE<spacedim>&				global_data,
																const double									D,
																const double									c0,
																const double									alpha,
																const unsigned int								method,
																const unsigned int								sym_mode,
																const unsigned int								formulation)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DissipationSpeciesFlux00", 1, 1),
global_data(&global_data),
formulation(formulation),
D(D),
c0(c0),
method(method),
sym_mode(sym_mode),
alpha(alpha)
{
	if(method == 0)
		Assert(sym_mode == 0, ExcMessage("If discretization method 0 is used, sym_mode must also be 0!"));
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
DissipationSpeciesFlux00<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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
	Vector<double> i(3);
	if(formulation == 0)
		for(unsigned int m = 0; m < 3; ++m)
			i[m] = (e_omega[m] - e_omega_ref_sets[0][m])/delta_t;
	else
		for(unsigned int m = 0; m < 3; ++m)
			i[m] = e_omega[m]/delta_t;

	double c_total = 0.0;
	if(method == 0)
		c_total = c0 + e_omega_ref_sets[0][3];
	else if(method == 1)
		c_total = c0 + alpha * e_omega[3] + (1. - alpha) * e_omega_ref_sets[0][3];
	else if(method == 2)
	{
		if(global_data->get_predictor_step())
		{
			c_total = c0 + e_omega_ref_sets[0][3];
			hidden_vars[0] = e_omega[3];
		}
		else
			c_total = c0 + alpha * hidden_vars[0] + (1. - alpha) * e_omega_ref_sets[0][3];
	}
	else
		Assert(false, ExcMessage("Requested a method for time integration of dissipation which is not implemented!"));

	///////////////////////////////////
	// method 0 for time integration //
	///////////////////////////////////
	if(method == 0)
	{
		if(get<0>(requested_quantities))
			h_omega = delta_t/(2.0*D*c_total)*(i*i);

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_1[m] = 1.0/(D*c_total) * i[m];
		}

		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_2(m, m) = 1.0/(delta_t*D*c_total);
		}
	}

	///////////////////////////////////
	// method 1 for time integration //
	///////////////////////////////////
	else if(method == 1)
	{
		if(get<0>(requested_quantities))
			h_omega=0.0;

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_1[m] = 1.0/(D*c_total) * i[m];
		}
	
		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
			{
				h_omega_2(m, m) = 1.0/(delta_t*D*c_total);
				if(sym_mode == 0)
					h_omega_2(m, 3) = -i[m]/(D*c_total*c_total)*alpha;
			}
		}
	}
	///////////////////////////////////
	// method 2 for time integration //
	///////////////////////////////////
	else if(method == 2)
	{
		if(get<0>(requested_quantities))
			h_omega = delta_t/(2.0*D*c_total)*(i*i);

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_1[m] = 1.0/(D*c_total) * i[m];
			h_omega_1[3] = 0.0;
		}

		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
			{
				h_omega_2(m, m) = 1.0/(delta_t*D*c_total);
				h_omega_2(m, 3) = 0.0;
				h_omega_2(3, m) = 0.0;
			}
			h_omega_2(3,3) = 0.0;
		}
	}

	return false;
}

template<unsigned int spacedim>
double
DissipationSpeciesFlux00<spacedim>::get_maximum_step(	const Vector<double>& 			e_omega,
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

template class DissipationSpeciesFlux00<2>;
template class DissipationSpeciesFlux00<3>;
