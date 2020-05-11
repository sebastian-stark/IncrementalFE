#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>
#include <incremental_fe/scalar_functionals/penalty_00.h>

#include <math.h>


using namespace dealii;
using namespace std;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
Penalty00<spacedim>::Penalty00(	const vector<DependentField<spacedim, spacedim>>	e_omega,
								const set<dealii::types::material_id>				domain_of_integration,
								const Quadrature<spacedim>							quadrature,
								GlobalDataIncrementalFE<spacedim>&					global_data,
								const double										mu,
								const double										c0,
								const double										alpha,
								const unsigned int								method):
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "Penalty00", 1),
global_data(&global_data),
mu(mu),
c0(c0),
alpha(alpha),
method(method)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
Penalty00<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
									const vector<Vector<double>>&	e_omega_ref_sets,
									Vector<double>&					/*hidden_vars*/,
									const Point<spacedim>&			x,
									double&							h_omega,
									Vector<double>&					h_omega_1,
									FullMatrix<double>&				h_omega_2,
									const tuple<bool, bool, bool>	requested_quantities)
const
{
	Assert(e_omega.size() == this->e_omega.size(), ExcMessage("Called get_h_omega with invalid size of e vector!"));
	Assert(e_omega_ref_sets.size() >= this->n_ref_sets, ExcMessage("Called get_h_omega with not enough datasets for the reference values of the independent fields!"));
	Assert(e_omega_ref_sets[0].size() == this->e_omega.size(), ExcMessage("Called get_h_omega with invalid size of e_ref vector!"));

	const double mu_ = mu;

	//no negative concentrations allowed -> if negative concentration return with error
	if(e_omega[0] < 0.0)
	{
		global_data->write_error_message("Negative species concentration in " + this->name, x);
		return true;
	}

	if(method == 0)
	{

		//use alpha=1 in first step to equilibrate solution
		const double alpha_ = ( global_data->get_time_step() == 1 ? 1. : alpha);

		const double c_alpha = alpha_*e_omega[0] + (1. - alpha_)*e_omega_ref_sets[0][0];

		if(get<0>(requested_quantities))
			h_omega = - mu_ * log(c_alpha/c0) / alpha_;

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			h_omega_1[0] = -mu_/c_alpha;
		}

		if(get<2>(requested_quantities))
			h_omega_2(0,0) = mu_/c_alpha/c_alpha*alpha_;
	}
	else
	{
		if(get<0>(requested_quantities))
			h_omega = - mu_ * log(e_omega[0]/c0);

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size()!=this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			h_omega_1[0] = -mu_/e_omega[0];
		}
		if(get<2>(requested_quantities))
			h_omega_2(0,0) = mu_/e_omega[0]/e_omega[0];

		if( (alpha != 1.) && (global_data->get_time_step() != 1) )
		{
			//gradient at reference state
			Vector<double>	h_omega_1_ref;
			if(get<0>(requested_quantities) || get<1>(requested_quantities))
			{
				h_omega_1_ref.reinit(e_omega.size());
				h_omega_1_ref[0] = -mu_/e_omega_ref_sets[0][0];
			}

			if(get<0>(requested_quantities))
				h_omega = h_omega * alpha + h_omega_1_ref[0] * e_omega[0] * (1. - alpha);

			if(get<1>(requested_quantities))
				h_omega_1[0] = h_omega_1[0] * alpha + h_omega_1_ref[0] * (1. - alpha);;

			if(get<2>(requested_quantities))
				h_omega_2(0,0) = h_omega_2(0,0) * alpha;
		}
	}

	return false;
}


template class Penalty00<2>;
template class Penalty00<3>;
