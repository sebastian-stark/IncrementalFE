#include <incremental_fe/scalar_functionals/chemical_potential_01.h>

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
ChemicalPotential01<spacedim>::ChemicalPotential01(	const vector<DependentField<spacedim,spacedim>>	e_omega,
													const set<types::material_id>					domain_of_integration,
													const Quadrature<spacedim>						quadrature,
													GlobalDataIncrementalFE<spacedim>&				global_data,
													const double									RT,
													const double									c0,
													const double									mu0,
													const double									alpha,
													const double									c_th_c0):
ScalarFunctional<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, "ChemicalPotential01", 1),
global_data(&global_data),
RT(RT),
c0(c0),
mu0(mu0),
alpha(alpha),
c_th_c0(c_th_c0),
log_c_th_c0(log(c_th_c0))
{
	Assert( (alpha >= 0.0) && (alpha <= 1.0), ExcMessage("alpha must be in the range 0 <= alpha <= 1 !"));
}

template<unsigned int spacedim>
bool
ChemicalPotential01<spacedim>::get_h_omega(	const Vector<double>&			e_omega,
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

	//compute beforehand because this may be used twice and is an expensive operation
	const double log_c_c0 = log(e_omega[0]/c0);

	Vector<double> F(9);
	for(unsigned int m = 0; m < 9; ++m)
		F[m] = e_omega[m + 1];

	const double c = e_omega[0];
	const double c_ref = e_omega_ref_sets[0][0];

	const double J = get_J(F);
	Assert( J > 0, ExcMessage("The determinant of the deformation gradient must be larger than zero"));

	Vector<double> dJ_dF;
	FullMatrix<double> d2J_dF2;
	if(get<1>(requested_quantities) || (get<2>(requested_quantities)))
	{
		dJ_dF.reinit(9);
		get_dJ_dF(F, dJ_dF);
	}
	if(get<2>(requested_quantities))
	{
		d2J_dF2.reinit(9, 9);
		get_d2J_dF2(F, d2J_dF2);
	}

	//compute derivatives at alpha = 1.0
	if(get<0>(requested_quantities))
	{
		if(c/c0 < c_th_c0)
			h_omega = RT * ( c_th_c0*c0 * ( log_c_th_c0 * (log_c_c0 - log_c_th_c0 + 1.) - 1. ) + mu0 * c );
		else
			h_omega = RT * c * (log_c_c0 + mu0 - 1.0);

		//large def part
		h_omega -= RT * (c - c0) * log(J);
	}

	if(get<1>(requested_quantities))
	{
		if(h_omega_1.size() != this->e_omega.size())
			h_omega_1.reinit(this->e_omega.size());

		if(c/c0 < c_th_c0)
			h_omega_1[0] = RT * ( c_th_c0*c0/c*log_c_th_c0 + mu0 );
		else
			h_omega_1[0] = RT * (log_c_c0 + mu0);

		//large def part
		h_omega_1[0] -= RT * log(J);
		for(unsigned int m = 0; m < 9; ++m)
			h_omega_1[m + 1] = -RT * (c - c0) / J * dJ_dF[m];
	}
	if(get<2>(requested_quantities))
	{
		if( (h_omega_2.size()[0] != this->e_omega.size()) || (h_omega_2.size()[1] != this->e_omega.size()) )
			h_omega_2.reinit(this->e_omega.size(), this->e_omega.size());

		if(c/c0<c_th_c0)
			h_omega_2(0,0) = RT/c * (-c_th_c0*c0/c*log_c_th_c0);
		else
			h_omega_2(0,0) = RT/c;

		//large def part
		for(unsigned int m = 0; m < 9; ++m)
		{
			h_omega_2(0, m + 1) = h_omega_2(m + 1, 0) = -RT / J * dJ_dF[m];
			for(unsigned int n = 0; n < 9; ++n)
				h_omega_2(m + 1, n + 1) = RT * (c - c0) / J * ( 1.0 / J * dJ_dF[m] * dJ_dF[n] - d2J_dF2(m, n) );
		}
	}

	//now weight derivatives according to alpha
	if (alpha != 1.)
	{
		//gradient at reference state
		Vector<double>	h_omega_1_ref;
		if(get<0>(requested_quantities) || get<1>(requested_quantities))
		{
			Vector<double> F_ref(9);
			for(unsigned int m = 0; m < 9; ++m)
				F_ref[m] = e_omega_ref_sets[0][m + 1];

			const double J_ref = get_J(F_ref);
			Assert( J_ref > 0, ExcMessage("The determinant of the deformation gradient in the reference state must be larger than zero"));

			Vector<double> dJ_dF_ref(9);
			get_dJ_dF(F_ref, dJ_dF_ref);

			h_omega_1_ref.reinit(e_omega.size());
			if(c_ref/c0 < c_th_c0)
				h_omega_1_ref[0] = RT * ( c_th_c0*c0/c_ref*log_c_th_c0 + mu0 );
			else
				h_omega_1_ref[0] = RT * (log(c_ref/c0) + mu0);

			//large def part
			h_omega_1_ref[0] -= RT * log(J_ref);
			for(unsigned int m = 0; m < 9; ++m)
				h_omega_1_ref[m + 1] = -RT * (c_ref - c0) / J_ref * dJ_dF_ref[m];
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
ChemicalPotential01<spacedim>::get_maximum_step(const Vector<double>& 			e_omega,
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


template class ChemicalPotential01<2>;
template class ChemicalPotential01<3>;
