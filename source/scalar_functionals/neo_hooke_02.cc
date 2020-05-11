#include <incremental_fe/scalar_functionals/neo_hooke_02.h>

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
NeoHooke02<spacedim>::compute_derivatives(	const Vector<double>&			e_omega,
											double&							val,
											Vector<double>&					d1,
											FullMatrix<double>& 			d2,
											const tuple<bool, bool, bool>&	requested_quantities,
											const dealii::Point<spacedim>& 	x)
const
{
	const double mu_ = scaling_function.value(x) * mu;
	const double lambda_ = scaling_function.value(x) * lambda;

	//deformation gradient
	Vector<double> F(9);
	for(unsigned int m = 0; m < 9; ++m)
		F[m] = e_omega[m];

	//I_1_e, J_e
	const double I_1_e = F * F;
	const double J_e = get_J(F);
	Assert(J_e > 0, ExcMessage("The determinant of the elastic strain must be greater than zero"));

	//compute value of potential
	if(get<0>(requested_quantities))
		val = 0.5 * mu_ * (I_1_e - 3.0 - 2.0 * log(J_e)) + 0.5 * lambda_ * log(J_e) * log(J_e);

	//if no derivatives requested, we can return here
	if( (!get<1>(requested_quantities)) && (!get<2>(requested_quantities)) )
		return;

	//derivatives of I_1_e
	Vector<double> dI_1_e(9);
	FullMatrix<double> d2I_1_e;
	for(unsigned int m = 0; m < 9; ++m)
		dI_1_e[m] = 2.0 * F[m];
	if(get<2>(requested_quantities))
	{
		d2I_1_e.reinit(9,9);
		for(unsigned int m = 0; m < 9; ++m)
			d2I_1_e(m,m) = 2.0;
	}

	//derivatives of J_e
	Vector<double> dJ_e(9);
	FullMatrix<double> d2J_e;
	get_dJ_dF(F, dJ_e);
	if(get<2>(requested_quantities))
	{
		d2J_e.reinit(9,9);
		get_d2J_dF2(F, d2J_e);
	}

	//first derivatives of potential w.r.t. J_e and I_1_e
	const double dpsi_dI_1_e = 0.5 * mu_;
	const double dpsi_dJ_e = -mu_/J_e + lambda_ * log(J_e)/J_e;

	//first derivative
	if(get<1>(requested_quantities))
	{
		d1.reinit(9);
		for(unsigned int m = 0; m < 9; ++m)
			d1[m] = dpsi_dI_1_e * dI_1_e[m] + dpsi_dJ_e * dJ_e[m];
	}

	//second derivative
	if(get<2>(requested_quantities))
	{
		const double d2psi_dJ_e2 = mu_/J_e/J_e + lambda_ * (1.0/J_e/J_e) * (1.0 - log(J_e));
		d2.reinit(9,9);
		for(unsigned int m = 0; m < 9; ++m)
			for(unsigned int n = 0; n < 9; ++n)
				d2(m, n) = dpsi_dI_1_e * d2I_1_e(m, n) + dpsi_dJ_e * d2J_e(m, n) + d2psi_dJ_e2 * dJ_e[m] * dJ_e[n];
	}
}

template<unsigned int spacedim>
NeoHooke02<spacedim>::NeoHooke02(	const vector<DependentField<spacedim,spacedim>>	e_omega,
									const set<types::material_id>					domain_of_integration,
									const Quadrature<spacedim>						quadrature,
									GlobalDataIncrementalFE<spacedim>&				global_data,
									const double									lambda,
									const double									mu,
									const double									alpha,
									const Function<spacedim>&						scaling_function)
:
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "NeoHooke02", 1),
global_data(&global_data),
lambda(lambda),
mu(mu),
alpha(alpha),
scaling_function(scaling_function)
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
NeoHooke02<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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

	//compute derivatives at alpha = 1.0
	compute_derivatives(e_omega, h_omega, h_omega_1, h_omega_2, requested_quantities, x);

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		Vector<double>	h_omega_1_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			h_omega_1_ref.reinit(e_omega.size());
			compute_derivatives(e_omega_ref_sets[0], h_omega, h_omega_1_ref, h_omega_2, make_tuple(false, true, false), x);
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
NeoHooke02<spacedim>::get_maximum_step(	const Vector<double>& 			e_omega,
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

template class NeoHooke02<2>;
template class NeoHooke02<3>;
