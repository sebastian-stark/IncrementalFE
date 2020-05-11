#include <incremental_fe/scalar_functionals/neo_hooke_01.h>

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

	void get_d2J_dF2(	const Vector<double>& F,
						FullMatrix<double>& d2J_dF2)
	{
		d2J_dF2(4,0) =  F[8];
		d2J_dF2(5,0) = -F[7];
		d2J_dF2(7,0) = -F[5];
		d2J_dF2(8,0) =  F[4];

		d2J_dF2(3,1) = -F[8];
		d2J_dF2(5,1) =  F[6];
		d2J_dF2(6,1) =  F[5];
		d2J_dF2(8,1) = -F[3];

		d2J_dF2(3,2) =  F[7];
		d2J_dF2(4,2) = -F[6];
		d2J_dF2(6,2) = -F[4];
		d2J_dF2(7,2) =  F[3];

		d2J_dF2(1,3) = -F[8];
		d2J_dF2(2,3) =  F[7];
		d2J_dF2(7,3) =  F[2];
		d2J_dF2(8,3) = -F[1];

		d2J_dF2(0,4) =  F[8];
		d2J_dF2(2,4) = -F[6];
		d2J_dF2(6,4) = -F[2];
		d2J_dF2(8,4) =  F[0];

		d2J_dF2(0,5) = -F[7];
		d2J_dF2(1,5) =  F[6];
		d2J_dF2(6,5) =  F[1];
		d2J_dF2(7,5) = -F[0];

		d2J_dF2(1,6) =  F[5];
		d2J_dF2(2,6) = -F[4];
		d2J_dF2(4,6) = -F[2];
		d2J_dF2(5,6) =  F[1];

		d2J_dF2(0,7) = -F[5];
		d2J_dF2(2,7) =  F[3];
		d2J_dF2(3,7) =  F[2];
		d2J_dF2(5,7) = -F[0];

		d2J_dF2(0,8) =  F[4];
		d2J_dF2(1,8) = -F[3];
		d2J_dF2(3,8) = -F[1];
		d2J_dF2(4,8) =  F[0];
	}
}


template<unsigned int spacedim>
void
NeoHooke01<spacedim>::compute_derivatives(	const Vector<double>&			e_omega,
											double&							val,
											Vector<double>&					d1,
											FullMatrix<double>& 			d2,
											const tuple<bool, bool, bool>&	requested_quantities)
const
{
	//deformation gradient
	Vector<double> F(9);
	for(unsigned int m = 0; m < 9; ++m)
		F[m] = e_omega[m];

	const double J = get_J(F);
	Assert(J > 0, ExcMessage("The determinant of the deformation gradient must be greater than zero"));

	double tr_C = F*F;
	//compute value of potential
	if(get<0>(requested_quantities))
		val = 0.5 * mu * (tr_C - 3.0) - mu * log(J) + lambda * (1.0-n_0) * (1.0-n_0) * ( (J-1.0)/(1.0-n_0) - log((J-n_0)/(1.0-n_0)) );

	//if no derivatives requested, we can return here
	if( (!get<1>(requested_quantities)) && (!get<2>(requested_quantities)) )
		return;

	Vector<double> dJ_dF(9);
	FullMatrix<double> d2J_dF2;
	get_dJ_dF(F, dJ_dF);
	if(get<2>(requested_quantities))
	{
		d2J_dF2.reinit(9,9);
		get_d2J_dF2(F, d2J_dF2);
	}

	//first derivative
	if(get<1>(requested_quantities))
	{
		d1.reinit(9);
		for(unsigned int m = 0; m < 9; ++m)
			d1[m] = mu * ( F[m] - 1.0/J * dJ_dF[m] ) + lambda * (1.0-n_0) * (J-1.0) / (J-n_0) * dJ_dF[m];
	}

	//second derivative
	if(get<2>(requested_quantities))
	{
		d2.reinit(9,9);
		for(unsigned int m = 0; m < 9; ++m)
			for(unsigned int n = 0; n < 9; ++n)
				d2(m, n) = mu * ( 1.0/J/J * dJ_dF[m] * dJ_dF[n] - 1.0/J * d2J_dF2(m,n)) + lambda * (1.0-n_0)/(J-n_0) * ( ( 1.0 - (J-1.0)/(J-n_0) ) * dJ_dF[m] * dJ_dF[n] + (J-1.0) * d2J_dF2(m,n) );

		for(unsigned int m = 0; m < 9; ++m)
			d2(m, m) += mu;

	}
}

template<unsigned int spacedim>
NeoHooke01<spacedim>::NeoHooke01(	const vector<DependentField<spacedim,spacedim>>	e_omega,
									const set<types::material_id>					domain_of_integration,
									const Quadrature<spacedim>						quadrature,
									GlobalDataIncrementalFE<spacedim>&				global_data,
									const double									lambda,
									const double									mu,
									const double									n_0,
									const double									alpha)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "NeoHooke01", 1),
global_data(&global_data),
lambda(lambda),
mu(mu),
n_0(n_0),
alpha(alpha)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
NeoHooke01<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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
NeoHooke01<spacedim>::get_maximum_step(	const Vector<double>& 			e_omega,
										const vector<Vector<double>>&	/*e_omega_ref_sets*/,
										const Vector<double>& 			delta_e_omega,
										const Vector<double>& 			/*hidden_vars*/,
										const Point<spacedim>& 			/*x*/)

const
{
	double factor = 2.0;
	Vector<double> e(e_omega.size());

	while(true)
	{
		for(unsigned int m = 0; m < e.size(); ++m)
			e[m] = e_omega[m] + factor * delta_e_omega[m];
		if(get_J(e) > 0.0)
			return factor;

		factor *= 0.5;
		Assert(factor > 0.0, ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient stays positive!"));
	}

	return factor;
}

template class NeoHooke01<2>;
template class NeoHooke01<3>;
