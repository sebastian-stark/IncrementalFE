#include <incremental_fe/scalar_functionals/chemical_potential_03.h>

#include <math.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

namespace
{
	double get_J(const Vector<double>& F)
	{
		return	  F[0] * F[4] * F[8]
				+ F[1] * F[5] * F[6]
				+ F[2] * F[3] * F[7]
				- F[2] * F[4] * F[6]
				- F[1] * F[3] * F[8]
				- F[0] * F[5] * F[7];
	}

	void get_dJ_dF(	const Vector<double>& 	F,
					Vector<double>& 		dJ_dF)
	{
		dJ_dF[0] = F[4] * F[8] - F[5] * F[7];
		dJ_dF[1] = F[5] * F[6] - F[3] * F[8];
		dJ_dF[2] = F[3] * F[7] - F[4] * F[6];
		dJ_dF[3] = F[2] * F[7] - F[1] * F[8];
		dJ_dF[4] = F[0] * F[8] - F[2] * F[6];
		dJ_dF[5] = F[1] * F[6] - F[0] * F[7];
		dJ_dF[6] = F[1] * F[5] - F[2] * F[4];
		dJ_dF[7] = F[2] * F[3] - F[0] * F[5];
		dJ_dF[8] = F[0] * F[4] - F[1] * F[3];
	}
}

template<unsigned int spacedim>
void
ChemicalPotential03<spacedim>::compute_derivatives(	const Vector<double>&	e_omega,
											double&							val,
											Vector<double>&					d1,
											FullMatrix<double>& 			d2,
											const tuple<bool, bool, bool>&	requested_quantities)
const
{
	const double c = e_omega[0];
	//deformation gradient
	Vector<double> F(9);
	for(unsigned int m = 0; m < 9; ++m)
		F[m] = e_omega[m + 1];

	const double J = get_J(F);


	//compute value of potential
	if(get<0>(requested_quantities))
		val = 0.0;

	//if no derivatives requested, we can return here
	if( (!get<1>(requested_quantities)) && (!get<2>(requested_quantities)) )
		return;

	//first derivative
	if(get<1>(requested_quantities))
	{
		d1.reinit(e_omega.size());
		d1[0] = mu_0 + RT * log(c) + RT * log(V_m_f/J);
	}

	//second derivative
	if(get<2>(requested_quantities))
	{
		d2.reinit(e_omega.size(), e_omega.size());
		Vector<double> dJ_dF(9);
		get_dJ_dF(F, dJ_dF);
		d2(0,0) = RT/c;
		for(unsigned int m = 0; m < 9; ++m)
			d2(0, m+1) = -RT/J * dJ_dF[m];
	}
}

template<unsigned int spacedim>
ChemicalPotential03<spacedim>::ChemicalPotential03(	const vector<GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>									domain_of_integration,
													const Quadrature<spacedim>										quadrature,
													GlobalDataIncrementalFE<spacedim>&								global_data,
													const double													RT,
													const double													mu_0,
													const double													V_m_f,
													const double													alpha)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "ChemicalPotential03", 1),
global_data(&global_data),
RT(RT),
mu_0(mu_0),
V_m_f(V_m_f),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
ChemicalPotential03<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
											const vector<Vector<double>>&	e_omega_ref_sets,
											Vector<double>&					/*hidden_vars*/,
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

	//compute derivatives at alpha = 1.0
	compute_derivatives(e_omega, h_omega, h_omega_1, h_omega_2, requested_quantities);

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		Vector<double>	h_omega_1_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_omega_1_ref.reinit(e_omega.size());
			compute_derivatives(e_omega_ref_sets[0], h_omega, h_omega_1_ref, h_omega_2, make_tuple(false, true, false));
		}

		if(get<0>(requested_quantities))
			h_omega = 0.0;

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
ChemicalPotential03<spacedim>::get_maximum_step(const Vector<double>& 			e_omega,
												const vector<Vector<double>>&	/*e_omega_ref_sets*/,
												const Vector<double>& 			delta_e_omega,
												const Vector<double>& 			/*hidden_vars*/,
												const Point<spacedim>& 			/*x*/)

const
{
	const double max_step_1 = - e_omega[0] / delta_e_omega[0];
	if( (isnan(max_step_1) || (max_step_1 < 0.0)) )
		return DBL_MAX;
	else
		return max_step_1;
}

template class ChemicalPotential03<2>;
template class ChemicalPotential03<3>;
