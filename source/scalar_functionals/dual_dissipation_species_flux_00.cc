#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <incremental_fe/scalar_functionals/dual_dissipation_species_flux_00.h>

using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
DualDissipationSpeciesFlux00<spacedim>::DualDissipationSpeciesFlux00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
																		const set<types::material_id>					domain_of_integration,
																		const Quadrature<spacedim>						quadrature,
																		GlobalDataIncrementalFE<spacedim>&				global_data,
																		const double									D,
																		const double									c0,
																		const double									alpha,
																		const unsigned int								method,
																		const unsigned int								sym_mode)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "DualDissipationSpeciesFlux00", 1, 1),
global_data(&global_data),
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
DualDissipationSpeciesFlux00<spacedim>::get_h_omega(const Vector<double>&			e_omega,
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

	//potential
	const double eta = e_omega[4];
	const double c = e_omega[3];
	const double c_ref = e_omega_ref_sets[0][3];

	//average flux during time step
	Vector<double> f(3);
	for(unsigned int m = 0; m < 3; ++m)
		f[m] = e_omega[m];

	double c_total;
	if(method < 2)
		c_total = c0 + alpha*c + (1. - alpha)*c_ref;
	else
	{
		if(global_data->get_predictor_step())
		{
			c_total = c0 + c_ref;
			hidden_vars[0] = c;
		}
		else
		{
			c_total = c0 + alpha * hidden_vars[0] + (1. - alpha)*c_ref;
		}
	}

	///////////////////////////////////
	// method 0 for time integration //
	///////////////////////////////////
	if(method == 0)
	{
		if(get<0>(requested_quantities))
			h_omega =  -1.0 * ( delta_t * D*c_total/(2.0)*(f*f) + eta * (c-c_ref) );

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_1[m] = -delta_t * (D*c_total * f[m]);
			h_omega_1[3] = -1.0 * ( delta_t * D*alpha/(2.0)*(f*f) + eta);
			h_omega_1[4] = -1.0 * (c-c_ref);
		}

		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
			{
				h_omega_2(m, m) = -delta_t * D*c_total;
				h_omega_2(m, 3) = -delta_t * D*alpha*f[m];
				h_omega_2(3, m) = h_omega_2(m,3);
			}
			h_omega_2(3,4) = -1.0;
			h_omega_2(4,3) = -1.0;
		}
	}
	///////////////////////////////////
	// method 1 for time integration //
	///////////////////////////////////
	else if(method == 1)
	{
		if(get<0>(requested_quantities))
			h_omega = 0.0;

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_1[m] = -delta_t * D * c_total * f[m];
			h_omega_1[3] = -1.0 * eta;
			h_omega_1[4] = -1.0 * (c-c_ref);
		}
	
		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
			{
				h_omega_2(m, m) = -delta_t * D*c_total;
				if(sym_mode == 0)
					h_omega_2(m, 3) = -delta_t * D* alpha * f[m];
			}
			h_omega_2(3,4) = -1.0;
			h_omega_2(4,3) = -1.0;
		}
	}
	///////////////////////////////////
	// method 2 for time integration //
	///////////////////////////////////
	else
	{
		if(get<0>(requested_quantities))
			h_omega = -1.0 * ( delta_t * D*c_total/(2.0)*(f*f) + eta * (c-c_ref) );

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
				h_omega_1[m] = -delta_t * D* c_total * f[m];
			h_omega_1[3] = -1.0 * eta;
			h_omega_1[4] = -1.0 * (c-c_ref);
		}

		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			for(unsigned int m = 0; m < 3; ++m)
			{
				h_omega_2(m, m) = -delta_t*D*c_total;
				h_omega_2(m, 3) = 0.0;
				h_omega_2(3, m) = 0.0;
			}
			h_omega_2(3,4) = -1.0;
			h_omega_2(4,3) = -1.0;
		}
	}

	return false;
}

template<unsigned int spacedim>
double
DualDissipationSpeciesFlux00<spacedim>::get_maximum_step(	const Vector<double>& 			e_omega,
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

template class DualDissipationSpeciesFlux00<2>;
template class DualDissipationSpeciesFlux00<3>;
