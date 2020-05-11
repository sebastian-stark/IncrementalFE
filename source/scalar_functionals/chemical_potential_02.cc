#include <incremental_fe/scalar_functionals/chemical_potential_02.h>

#include <math.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
ChemicalPotential02<spacedim>::ChemicalPotential02(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									RT,
													const double									mu_0,
													const double									alpha)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "ChemicalPotential02", 1),
global_data(&global_data),
RT(RT),
mu_0(mu_0),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
ChemicalPotential02<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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
	if(e_omega[1] <= 0.0)
	{
		this->global_data->write_error_message("Negative species concentration in " + this->name, x);
		return true;
	}

	const double c = e_omega[0];
	const double c_F = e_omega[1];
	const double c_ref = e_omega_ref_sets[0][0];
	const double c_F_ref = e_omega_ref_sets[0][1];

	//compute beforehand because this may be used twice and is an expensive operation
	const double log_c_c_F = log(c/c_F);
	const double log_c_ref_c_F_ref = log(c_ref/c_F_ref);

	//compute derivatives at alpha = 1.0
	if(get<0>(requested_quantities))
	{
		h_omega = mu_0 * c + RT * c * (log_c_c_F - 1.0);
	}

	if(get<1>(requested_quantities))
	{
		if(h_omega_1.size() != this->e_omega.size())
			h_omega_1.reinit(this->e_omega.size());
		h_omega_1[0] = mu_0 + RT * log_c_c_F;
		h_omega_1[1] = -RT * c/c_F;
	}
	if(get<2>(requested_quantities))
	{
		if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
			h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());
		h_omega_2(0,0) = RT/c;
		h_omega_2(1,1) = RT * c / c_F / c_F;
		h_omega_2(0,1) = h_omega_2(1,0) = -RT / c_F;
	}

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		Vector<double>	h_omega_1_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_omega_1_ref.reinit(e_omega.size());
			h_omega_1_ref[0] = mu_0 + RT * log_c_ref_c_F_ref;
			h_omega_1_ref[1] = -RT * c_ref/c_F_ref;
		}

		if(get<0>(requested_quantities))
			h_omega = h_omega * alpha + h_omega_1_ref * e_omega * (1. - alpha);

		if(get<1>(requested_quantities))
			for(unsigned int m = 0; m < e_omega.size(); ++m)
				h_omega_1[m] = h_omega_1[m] * alpha + h_omega_1_ref[m] * (1. - alpha);

		if(get<2>(requested_quantities))
			h_omega_2 *= alpha;
	}

	return false;
}

template<unsigned int spacedim>
double
ChemicalPotential02<spacedim>::get_maximum_step(const Vector<double>& 			e_omega,
												const vector<Vector<double>>&	/*e_omega_ref_sets*/,
												const Vector<double>& 			delta_e_omega,
												const Vector<double>& 			/*hidden_vars*/,
												const Point<spacedim>& 			/*x*/)

const
{
	const double max_step_1 = - e_omega[0] / delta_e_omega[0];
	const double max_step_2 = - e_omega[1] / delta_e_omega[1];
	if( (isnan(max_step_1) || (max_step_1 < 0.0)) && (isnan(max_step_2) || (max_step_2 < 0.0)) )
		return DBL_MAX;
	else
	{
		if(max_step_1 < 0.0)
			return max_step_2;
		else if(max_step_2 < 0.0)
			return max_step_1;
		else
			return std::min(max_step_1, max_step_2);
	}
}


template class ChemicalPotential02<2>;
template class ChemicalPotential02<3>;
