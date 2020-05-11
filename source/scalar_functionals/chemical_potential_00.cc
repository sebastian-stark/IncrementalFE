#include <incremental_fe/scalar_functionals/chemical_potential_00.h>

#include <math.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
ChemicalPotential00<spacedim>::ChemicalPotential00(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									RT,
													const double									c0,
													const double									mu0,
													const double									alpha,
													const unsigned int								method,
													const double									c_th_c0):
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "ChemicalPotential00", 1),
global_data(&global_data),
RT(RT),
c0(c0),
mu0(mu0),
alpha(alpha),
method(method),
c_th_c0(c_th_c0),
log_c_th_c0(log(c_th_c0))
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
ChemicalPotential00<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
											const vector<Vector<double>>&	e_omega_ref_sets,
											Vector<double>&					/*hidden_vars*/,
											const Point<spacedim>&			x,
											double&							h_omega,
											Vector<double>&					h_omega_1,
											FullMatrix<double>&				h_omega_2,
											const tuple<bool, bool, bool>	requested_quantities)
const
{

	Assert(e_omega.size() == this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e vector!"));
	Assert(e_omega_ref_sets.size() >= this->n_ref_sets,ExcMessage("Called get_h_omega with not enough datasets for the reference values of the independent fields!"));
	Assert(e_omega_ref_sets[0].size() == this->e_omega.size(),ExcMessage("Called get_h_omega with invalid size of e_omega_ref vector!"));

	//no negative concentrations allowed -> if negative concentration return with error
	if(e_omega[0] <= 0.0)
	{
		this->global_data->write_error_message("Negative species concentration in " + this->name, x);
		return true;
	}

	if(method == 0)
	{

		//use alpha=1. in first time step to equilibrate solution!
		const double alpha_ = alpha;

		//concentration at alpha
		const double c = (1. - alpha_)*e_omega_ref_sets[0][0] + alpha_*e_omega[0];

		//compute beforehand because this may be used twice and is an expensive operation
		const double log_c_c0 = log(c/c0);

		if(get<0>(requested_quantities))
		{
			if(c/c0 < c_th_c0)
				h_omega = RT * c * ( c_th_c0*c0/c * ( log_c_th_c0 * (log_c_c0 - log_c_th_c0 + 1.) - 1. ) + mu0 ) / alpha_;
			else
				h_omega = RT * c * (log_c_c0 + mu0 - 1.0) / alpha_;
		}

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			if(c/c0 < c_th_c0)
				h_omega_1[0] = RT * ( c_th_c0*c0/c*log_c_th_c0 + mu0 );
			else
				h_omega_1[0] = RT * (log_c_c0 + mu0);
		}
		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			if(c/c0 < c_th_c0)
				h_omega_2(0,0) = RT/c * (-c_th_c0*c0/c*log_c_th_c0) * alpha_;
			else
				h_omega_2(0,0) = RT/c * alpha_;
		}

	}
	else{

		//compute beforehand because this may be used twice and is an expensive operation
		const double log_c_c0 = log(e_omega[0]/c0);

		if(get<0>(requested_quantities))
		{
			if(e_omega[0]/c0 < c_th_c0)
				h_omega = RT * ( c_th_c0*c0 * ( log_c_th_c0 * (log_c_c0 - log_c_th_c0 + 1.) - 1. ) + mu0 * e_omega[0] );
			else
				h_omega = RT * e_omega[0] * (log_c_c0 + mu0 - 1.0);
		}

		if(get<1>(requested_quantities))
		{
			if(h_omega_1.size() != this->e_omega.size())
				h_omega_1.reinit(this->e_omega.size());
			if(e_omega[0]/c0 < c_th_c0)
				h_omega_1[0] = RT * ( c_th_c0*c0/e_omega[0]*log_c_th_c0 + mu0 );
			else
				h_omega_1[0] = RT * (log_c_c0 + mu0);
		}
		if(get<2>(requested_quantities))
		{
			if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
				h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
			if(e_omega[0]/c0<c_th_c0)
				h_omega_2(0,0) = RT/e_omega[0] * (-c_th_c0*c0/e_omega[0]*log_c_th_c0);
			else
				h_omega_2(0,0) = RT/e_omega[0];
		}

		if( (alpha != 1.) )
		{
			//gradient at reference state
			Vector<double>	h_omega_1_ref;
			if(get<0>(requested_quantities) || get<1>(requested_quantities))
			{
				h_omega_1_ref.reinit(e_omega.size());
				if(e_omega_ref_sets[0][0]/c0 < c_th_c0)
					h_omega_1_ref[0] = RT * ( c_th_c0*c0/e_omega_ref_sets[0][0]*log_c_th_c0 + mu0 );
				else
					h_omega_1_ref[0] = RT * (log(e_omega_ref_sets[0][0]/c0) + mu0);
			}

			if(get<0>(requested_quantities))
				h_omega = h_omega * alpha + h_omega_1_ref[0] * e_omega[0] * (1. - alpha);

			if(get<1>(requested_quantities))
				h_omega_1[0] = h_omega_1[0] * alpha + h_omega_1_ref[0] * (1. - alpha);

			if(get<2>(requested_quantities))
				h_omega_2(0,0) = h_omega_2(0,0) * alpha;

		}
	}

	return false;
}

template<unsigned int spacedim>
double
ChemicalPotential00<spacedim>::get_maximum_step(const Vector<double>& 			e_omega,
												const vector<Vector<double>>&	/*e_omega_ref_sets*/,
												const Vector<double>& 			delta_e_omega,
												const Vector<double>& 			/*hidden_vars*/,
												const Point<spacedim>& 			/*x*/)

const
{
	double max_step = - e_omega[0] / delta_e_omega[0];
	if(isnan(max_step) || (max_step < 0.0))
		return DBL_MAX;
	else
		return max_step;
}


template class ChemicalPotential00<2>;
template class ChemicalPotential00<3>;
