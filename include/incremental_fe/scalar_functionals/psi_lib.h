// --------------------------------------------------------------------------
// Copyright (C) 2020 by Sebastian Stark
//
// This file is part of the IncrementalFE library
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#ifndef INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_

#include <incremental_fe/scalar_functionals/psi.h>
#include <incremental_fe/fe_model.h>
#include <incremental_fe/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <limits>

namespace incrementalFE
{

namespace
{
	double get_J(	const dealii::Vector<double>& 	F,
					const bool						symmetric = false)
	{
		if(!symmetric)
		{
			return	  F[0] * F[4] * F[8]
					+ F[1] * F[5] * F[6]
					+ F[2] * F[3] * F[7]
					- F[2] * F[4] * F[6]
					- F[1] * F[3] * F[8]
					- F[0] * F[5] * F[7];
		}
		else
		{
			return 	  F[0] * F[3] * F[5]
					+ F[1] * F[2] * F[4] * 2.0
					- F[0]*F[4]*F[4]
					- F[1]*F[1]*F[5]
					- F[2]*F[2]*F[3];
		}
	}

	void get_dJ_dF(	const dealii::Vector<double>& 	F,
					dealii::Vector<double>& 		dJ_dF,
					const bool						symmetric = false)
	{
		if(!symmetric)
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
		else
		{
			dJ_dF[0] =        F[3] * F[5] -       F[4] * F[4];
			dJ_dF[1] = -2.0 * F[1] * F[5] + 2.0 * F[2] * F[4];
			dJ_dF[2] =  2.0 * F[1] * F[4] - 2.0 * F[2] * F[3];
			dJ_dF[3] =        F[0] * F[5] -       F[2] * F[2];
			dJ_dF[4] = -2.0 * F[0] * F[4] + 2.0 * F[1] * F[2];
			dJ_dF[5] =        F[0] * F[3] -       F[1] * F[1];
		}
	}

	void get_d2J_dF2(	const dealii::Vector<double>& 	F,
						dealii::FullMatrix<double>& 	d2J_dF2,
						const bool						symmetric = false)
	{
		if(!symmetric)
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
		else
		{
			d2J_dF2(0,3) =        F[5];
			d2J_dF2(0,4) = -2.0 * F[4];
			d2J_dF2(0,5) =        F[3];

			d2J_dF2(1,1) = -2.0 * F[5];
			d2J_dF2(1,2) =  2.0 * F[4];
			d2J_dF2(1,4) =  2.0 * F[2];
			d2J_dF2(1,5) = -2.0 * F[1];

			d2J_dF2(2,1) =  2.0 * F[4];
			d2J_dF2(2,2) = -2.0 * F[3];
			d2J_dF2(2,3) = -2.0 * F[2];
			d2J_dF2(2,4) =  2.0 * F[1];

			d2J_dF2(3,0) =        F[5];
			d2J_dF2(3,2) = -2.0 * F[2];
			d2J_dF2(3,5) =        F[0];

			d2J_dF2(4,0) = -2.0 * F[4];
			d2J_dF2(4,1) =  2.0 * F[2];
			d2J_dF2(4,2) =  2.0 * F[1];
			d2J_dF2(4,4) = -2.0 * F[0];

			d2J_dF2(5,0) =        F[3];
			d2J_dF2(5,1) = -2.0 * F[1];
			d2J_dF2(5,3) =        F[0];
		}
	}
}

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = RT c_0 h\left( \dfrac{a c + b}{c_0} \right)\f$,
 *
 * where
 *
 * \f$ h(x) = \begin{cases}
           x [ \ln(x)-1] \quad&\mathrm{if}\quad x>\epsilon\\
           \epsilon \{ \ln(\epsilon) [ \ln(x) - \ln(\epsilon) + 1] - 1\} \quad&\mathrm{else},
          \end{cases} \f$
 *
 * \f$R\f$ is the gas constant, \f$T\f$ the temperature, \f$c_0\f$ a
 * reference species concentration, \f$\mu_0\f$ a corresponding reference value
 * for the potential, \f$a\f$ and \f$b\f$ are constants, \f$c\f$ the species concentration, and \f$\epsilon \ll 1\f$ a regularization parameter
 * to avoid ill-conditioning if \f$c\f$ is too close to zero.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$
 */
template<unsigned int spacedim>
class PsiChemical00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * gas constant times absolute temperature
	 */
	const double
	RT;

	/**
	 * Reference concentration at which chemical potential is \f$RT\mu_0\f$
	 */
	const double
	c_0;

	/**
	 * \f$\mu_0\f$
	 */
	const double
	mu_0;

	/**
	 * \f$\epsilon\f$
	 */
	const double
	eps;

	/**
	 * \f$\ln\epsilon\f$
	 */
	const double
	log_eps;

	/**
	 * \f$a\f$
	 */
	const double
	a;

	/**
	 * \f$b\f$
	 */
	const double
	b;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		RT						PsiChemical00::RT
	 *
	 * @param[in]		c_0						PsiChemical00::c_0
	 *
	 * @param[in]		mu_0					PsiChemical00::mu_0
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		eps						PsiChemical00::eps
	 *
	 * @param[in]		a						PsiChemical00::a
	 *
	 * @param[in]		b						PsiChemical00::b
	 */
	PsiChemical00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																c_0,
					const double																mu_0,
					const double																alpha,
					const double																eps,
					const double																a = 1.0,
					const double																b = 0.0)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical00"),
	RT(RT),
	c_0(c_0),
	mu_0(mu_0),
	eps(eps),
	log_eps(log(eps)),
	a(a),
	b(b)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double c = a * values[0] + b;
		const double log_c_c_0 = log(c/c_0);
		const double c_th = eps*c_0;


		if( c <= 0.0 )
		{
			std::cout << "Negative concentration" << std::endl;
			return true;
		}


		if(get<0>(requested_quantities))
		{
			if(c/c_0 < eps)
				omega = RT *  c_0 * eps * ( log_eps * (log_c_c_0 - log_eps + 1.) - 1. )  + mu_0 * c;
			else
				omega = RT * c * (log_c_c_0 - 1.0) + mu_0 * c;
		}

		if(get<1>(requested_quantities))
		{
			if(c/c_0 < eps)
				d_omega[0] = (RT * eps * (c_0/c) * log_eps + mu_0) * a;
			else
				d_omega[0] = (RT * log_c_c_0 + mu_0) * a;
		}

		if(get<2>(requested_quantities))
		{
			if(c/c_0 < eps)
				d2_omega(0,0) = (- RT * eps * (c_0/c/c) * log_eps +  RT / c_th) * a * a;
			else
				d2_omega(0,0) = (RT / c) * a * a;
		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{
		double max_step = - 1.0 / delta_e_omega[0] * ( e_omega[0] + b / a );
		if(isnan(max_step) || (max_step <= 0.0))
			return numeric_limits<double>::max();
		else
			return max_step;
	}

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = \dfrac{a}{2} (c-b)^2\f$,
 *
 * where \f$a\f$ and \f$b\f$ are material parameters, and \f$c\f$ the species concentration.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$
 */
template<unsigned int spacedim>
class PsiChemical01 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$a\f$
	 */
	const double
	a;

	/**
	 * \f$b\f$
	 */
	const double
	b;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		a						PsiChemical01::a
	 *
	 * @param[in]		b						PsiChemical01::b
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiChemical01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																a,
					const double																b,
					const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical01"),
	a(a),
	b(b)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double c = values[0];

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * a * (c - b)  * (c - b);
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = a * (c - b);
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0,0) = a;
		}

		return false;
	}

};

/**
 * Class defining a domain related scalar functional such that
 *
 * \f$\dfrac{\mathrm{d} h^\Omega_\rho}{\mathrm{d} p} = p \dfrac{\mathrm{d}c}{\mathrm{d}p}(p) \f$,
 *
 * where \f$p\f$ is a state variable and \f$c(p)\f$ a function of this state variable.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$p\f$
 */
template<unsigned int spacedim>
class PsiTransformed00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * %Function \f$c(p)\f$
	 */
	const
	dealii::Function<1>&
	c_p;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		c_p						PsiTransformed00::c_p
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiTransformed00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const dealii::Function<1>&													c_p,
						const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiTransformed00"),
	c_p(c_p)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double p = values[0];
		const double dc_dp = (c_p.gradient(dealii::Point<1>(p)))[0];

		if(get<0>(requested_quantities))
		{
			omega = 0.0;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = p * dc_dp;
		}

		if(get<2>(requested_quantities))
		{
			const double d2c_dp2 = (c_p.hessian(dealii::Point<1>(p)))[0][0];
			d2_omega(0,0) = dc_dp + p * d2c_dp2;
		}

		return false;
	}

};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = 1/2 q A q + b q\f$,
 *
 * where \f$q\f$ is a state variable, and \f$A\f$ and \f$b\f$ are constants.
 * The implementation works also for vector valued \f$q\f$, in which case
 * \f$A\f$ is a matrix and \f$b\f$ a vector.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$q\f$
 */
template<unsigned int spacedim>
class PsiLinear00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * Matrix \f$A\f$
	 */
	const
	dealii::FullMatrix<double>
	A;

	/**
	 * %Vector \f$b\f$
	 */
	const
	dealii::Vector<double>
	b;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		A						PsiLinear00::A
	 *
	 * @param[in]		b						PsiLinear00::b
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiLinear00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const dealii::FullMatrix<double>&											A,
					const dealii::Vector<double>&												b,
					const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiLinear00"),
	A(A),
	b(b)
	{
		Assert( (A.m() == e_omega.size()), dealii::ExcMessage("The number of rows of A must coincide with the number of dependent fields!"));
		Assert( (A.n() == e_omega.size()), dealii::ExcMessage("The number of columns of A must coincide with the number of dependent fields!"));
		Assert( (b.size() == e_omega.size()), dealii::ExcMessage("The number of entries of b must coincide with the number of dependent fields!"));
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		dealii::Vector<double> A_q(b.size());
		A.vmult(A_q, values);

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * (values * A_q) + values * b;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < b.size(); ++m)
				d_omega[m] = A_q[m] + b[m];
		}

		if(get<2>(requested_quantities))
		{
			d2_omega = A;
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$h^\Sigma_\tau = 1/2 q A q + b q\f$,
 *
 * where \f$q\f$ is a state variable, and \f$A\f$ and \f$b\f$ are constants.
 * The implementation works also for vector valued \f$q\f$, in which case
 * \f$A\f$ is a matrix and \f$b\f$ a vector.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>	[0] \f$q\f$
 */
template<unsigned int spacedim>
class PsiLinearInterface00 : public incrementalFE::Psi<spacedim-1, spacedim>
{

private:

	/**
	 * Matrix \f$A\f$
	 */
	const
	dealii::FullMatrix<double>
	A;

	/**
	 * %Vector \f$b\f$
	 */
	const
	dealii::Vector<double>
	b;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Psi::global_data
	 *
	 * @param[in]		A						PsiLinear00::A
	 *
	 * @param[in]		b						PsiLinear00::b
	 *
	 * @param[in]		alpha					Psi::alpha
	 */
	PsiLinearInterface00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const dealii::FullMatrix<double>&												A,
							const dealii::Vector<double>&													b,
							const double																	alpha)
	:
	Psi<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, alpha, "PsiLinearInterface00"),
	A(A),
	b(b)
	{
		Assert( (A.m() == e_sigma.size()), dealii::ExcMessage("The number of rows of A must coincide with the number of dependent fields!"));
		Assert( (A.n() == e_sigma.size()), dealii::ExcMessage("The number of columns of A must coincide with the number of dependent fields!"));
		Assert( (b.size() == e_sigma.size()), dealii::ExcMessage("The number of entries of b must coincide with the number of dependent fields!"));
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		dealii::Vector<double> A_q(b.size());
		A.vmult(A_q, values);

		if(get<0>(requested_quantities))
		{
			sigma = 0.5 * (values * A_q) + values * b;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < b.size(); ++m)
				d_sigma[m] = A_q[m] + b[m];
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma = A;
		}

		return false;
	}

};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = \dfrac{\lambda}{2} \left[\mathrm{tr}\left(\boldsymbol{E}^\mathrm{e}\right)\right]^2 + \mu\mathrm{tr}\left[\left(\boldsymbol{E}^\mathrm{e}\right)^2\right]\f$,
 *
 * where
 *
 * \f$ \boldsymbol{E}^\mathrm{e} = \boldsymbol{E} - \dfrac{\Delta \varepsilon}{3} \left( \dfrac{c}{c^{\mathrm{ref}}} - 1 \right) \mathbf{I} \f$
 *
 * and
 *
 * \f$ \boldsymbol{E} = \dfrac{1}{2} ( \boldsymbol{F}^\top \cdot \boldsymbol{F} - \boldsymbol{I} ) \f$
 *
 * Here, \f$\boldsymbol{F}\f$ is the deformation gradient, \f$c\f$ is a species concentration, \f$\lambda\f$ and \f$\mu\f$ are Lame's constants, \f$\Delta \varepsilon\f$ describes the magnitude
 * of the isotropic stress free strain related to a change in \f$c\f$, and \f$c^{\mathrm{ref}}\f$ is the species concentration at which the stress free strain is zero.
 *
 * The relation represents a Saint Venant - Kirchhoff material with additional isotropic volumetric strain.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0]  \f$F_{xx}\f$ <br>
 * 																					[1]  \f$F_{xy}\f$ <br>
 * 																					[2]  \f$F_{xz}\f$ <br>
 * 																					[3]  \f$F_{yx}\f$ <br>
 * 																					[4]  \f$F_{yy}\f$ <br>
 * 																					[5]  \f$F_{yz}\f$ <br>
 * 																					[6]  \f$F_{zx}\f$ <br>
 * 																					[7]  \f$F_{zy}\f$ <br>
 * 																					[8]  \f$F_{zz}\f$ <br>
 * 																					[9]  \f$c\f$
 */
template<unsigned int spacedim>
class KirchhoffMaterial00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\mu\f$
	 */
	const double
	mu;

	/**
	 * \f$\Delta \varepsilon\f$
	 */
	const double
	deps;

	/**
	 * \f$c^\mathrm{ref}\f$
	 */
	const double
	c_ref;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		lambda					KirchhoffMaterial00::lambda
	 *
	 * @param[in]		mu						KirchhoffMaterial00::mu
	 *
	 * @param[in]		deps					KirchhoffMaterial00::deps
	 *
	 * @param[in]		c_ref					KirchhoffMaterial00::c_ref
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	KirchhoffMaterial00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const double																lambda,
						const double																mu,
						const double																deps,
						const double																c_ref,
						const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "KirchhoffMaterial00"),
	lambda(lambda),
	mu(mu),
	deps(deps),
	c_ref(c_ref)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		static dealii::Tensor<2, 3> F, P, T, E, I, B;
		for(unsigned int i = 0; i < 3; ++i)
			for(unsigned int j = 0; j < 3; ++j)
				F[i][j] = values[i*3 + j];
		const double c = values[9];

		I = dealii::unit_symmetric_tensor<3,double>();
		E = 0.5 * (dealii::contract<0, 0>(F, F) - I) - deps/3.0 * (c/c_ref - 1.0) * I;
		T = lambda * dealii::trace(E) * I + 2.0 * mu * E;

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * lambda * dealii::trace(E) * dealii::trace(E) + mu * trace(dealii::contract<0, 0>(E, E));
		}

		if(get<1>(requested_quantities))
		{

			P = dealii::contract<1,0>(F, T);
			for(unsigned int i = 0; i < 3; ++i)
				for(unsigned int j = 0; j < 3; ++j)
					d_omega[i*3 + j] = P[i][j];
			d_omega[9] = -deps/c_ref * trace(E) * (lambda + 2.0 * mu / 3.0);
		}

		if(get<2>(requested_quantities))
		{
			B = dealii::contract<1, 1>(F, F);
			for(unsigned int i = 0; i < 3; ++i)
				for(unsigned int j = 0; j < 3; ++j)
				{
					for(unsigned int k = 0; k < 3; ++k)
						for(unsigned int l = 0; l < 3; ++l)
							d2_omega[i*3 + j][k*3 + l] = I[i][k] * T[l][j] + lambda * F[i][j] * F[k][l] + mu * ( F[i][l] * F[k][j] + I[j][l] * B[k][i] );
					d2_omega[i*3 + j][9] = d2_omega[9][i*3 + j] = -deps/c_ref * (lambda + 2.0 * mu / 3.0) * F[i][j];
				}
			d2_omega[9][9] = deps/c_ref * deps/c_ref * (lambda + 2.0 * mu / 3.0);;
		}

		return false;
	}

};

/**
 *
 * Class defining Neo-Hooke material with "compression point" for hydrogel modeling.
 * See PhD thesis Acartürk (2009): Simulation of Charged Hydrated Porous Materials, Eq. (3.99)
 *
 * The integrand is
 *
 * \f$h^\Omega_\rho = \dfrac{\mu}{2} ( \mathrm{tr}\boldsymbol{C} - 3 ) - \mu \mathrm{ln}J + \lambda (1-n_0)^2 \left( \dfrac{J-1}{1-n_0} - \mathrm{ln}\dfrac{J-n_0}{1-n_0} \right) \f$,
 *
 * where
 *
 * \f$ \boldsymbol{C} =\boldsymbol{F}^\top \cdot \boldsymbol{F} \f$ is the right Cauchy-Green deformation tensor, \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$,
 * \f$n_0\f$ the initial volume fraction of the polymeric backbone, and \f$\mu\f$ and \f$\lambda\f$ Lame's constants.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0]  \f$F_{xx}\f$ <br>
 * 																					[1]  \f$F_{xy}\f$ <br>
 * 																					[2]  \f$F_{xz}\f$ <br>
 * 																					[3]  \f$F_{yx}\f$ <br>
 * 																					[4]  \f$F_{yy}\f$ <br>
 * 																					[5]  \f$F_{yz}\f$ <br>
 * 																					[6]  \f$F_{zx}\f$ <br>
 * 																					[7]  \f$F_{zy}\f$ <br>
 * 																					[8]  \f$F_{zz}\f$ <br>
 */
template<unsigned int spacedim>
class NeoHookeCompressionPoint00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\mu\f$
	 */
	const double
	mu;

	/**
	 * Initial volume fraction of polymeric backbone \f$n_0\f$
	 */
	const double
	n_0;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		lambda					NeoHookeCompressionPoint00::lambda
	 *
	 * @param[in]		mu						NeoHookeCompressionPoint00::mu
	 *
	 * @param[in]		n_0						NeoHookeCompressionPoint00::n_0
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	NeoHookeCompressionPoint00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																lambda,
								const double																mu,
								const double																n_0,
								const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "NeoHookeCompressionPoint00"),
	lambda(lambda),
	mu(mu),
	n_0(n_0)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		// deformation gradient and derived quantities
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m];

		const double J = get_J(F);
		Assert(J > 0, dealii::ExcMessage("The determinant of the deformation gradient must be greater than zero"));

		const double tr_C = F * F;

		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
		}

		// compute value of potential
		if(get<0>(requested_quantities))
			omega = 0.5 * mu * (tr_C - 3.0) - mu * log(J) + lambda * (1.0 - n_0) * (1.0 - n_0) * ( (J - 1.0) / (1.0 - n_0) - log( (J - n_0) / ( 1.0 - n_0) ) );


		// first derivative
		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = mu * ( F[m] - 1.0 / J * dJ_dF[m] ) + lambda * (1.0 - n_0) * (J - 1.0) / (J - n_0) * dJ_dF[m];
		}

		// second derivative
		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < 9; ++n)
					d2_omega(m, n) = mu * ( 1.0 / J / J * dJ_dF[m] * dJ_dF[n] - 1.0 / J * d2J_dF2(m,n) ) + lambda * (1.0 - n_0) / (J - n_0) * ( ( 1.0 - (J - 1.0) / (J - n_0) ) * dJ_dF[m] * dJ_dF[n] + (J - 1.0) * d2J_dF2(m,n) );
			for(unsigned int m = 0; m < 9; ++m)
				d2_omega(m, m) += mu;
		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{

		double factor = 2.0;
		dealii::Vector<double> e(e_omega.size());

		while(true)
		{
			for(unsigned int m = 0; m < e.size(); ++m)
				e[m] = e_omega[m] + factor * delta_e_omega[m];
			if(get_J(e) > n_0)
				return factor;

			factor *= 0.5;
			Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient stays positive!"));
		}

		return factor;
	}

};


/**
 *
 * Class defining a compressible Neo-Hooke material
 *
 * The integrand is
 *
 * \f$h^\Omega_\rho = \dfrac{\mu}{2} \left[ J^{2/3}_0\mathrm{tr}\boldsymbol{C} - 3 - 2 \mathrm{ln}(J J_0)  \right] + \dfrac{\lambda}{2} \left[\mathrm{ln}(J J_0)\right]^2\f$,
 *
 * where
 *
 * \f$ \boldsymbol{C} =\boldsymbol{F}^\top \cdot \boldsymbol{F} \f$ is the right Cauchy-Green deformation tensor, \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$,
 * \f$\mu\f$ and \f$\lambda\f$ Lame's constants, and \f$J_0\f$ a volumetric strain offset between the natural state of the material and the reference state.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0]  \f$F_{xx}\f$ <br>
 * 																					[1]  \f$F_{xy}\f$ <br>
 * 																					[2]  \f$F_{xz}\f$ <br>
 * 																					[3]  \f$F_{yx}\f$ <br>
 * 																					[4]  \f$F_{yy}\f$ <br>
 * 																					[5]  \f$F_{yz}\f$ <br>
 * 																					[6]  \f$F_{zx}\f$ <br>
 * 																					[7]  \f$F_{zy}\f$ <br>
 * 																					[8]  \f$F_{zz}\f$ <br>
 */
template<unsigned int spacedim>
class PsiNeoHooke00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\mu\f$
	 */
	const double
	mu;

	/**
	 * volumetric strain offset \f$J_0\f$
	 */
	const double
	J_0;

	/**
	 * %Function allowing to scale \f$\lambda\f$ and \f$\mu\f$ in dependence on position (the function must provide with the scaling factor)
	 */
	const dealii::Function<spacedim>&
	scaling_function;

	/**
	 * Global indices for which corresponding rows in the finite element systems are set to be zero
	 */
	set<unsigned int> const*
	ignore_dof_indices;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		lambda					PsiNeoHooke00::lambda
	 *
	 * @param[in]		mu						PsiNeoHooke00::mu
	 *
	 * @param[in]		scaling_function		PsiNeoHooke00::scaling_function
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		ignore_dof_indices		PsiNeoHooke00::ignore_dof_indices
	 *
	 * @param[in]		J_0						PsiNeoHooke00::J_0
	 */
	PsiNeoHooke00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																lambda,
					const double																mu,
					const dealii::Function<spacedim>&											scaling_function,
					const double																alpha,
					set<unsigned int> const*													ignore_dof_indices = nullptr,
					const double																J_0 = 1.0)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiNeoHooke00"),
	lambda(lambda),
	mu(mu),
	J_0(J_0),
	scaling_function(scaling_function),
	ignore_dof_indices(ignore_dof_indices)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		// scale mu and lambda
		const double mu_ = scaling_function.value(x) * mu;
		const double lambda_ = scaling_function.value(x) * lambda;

		// deformation gradient and derived quantities
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m];

		const double J_0_23 = pow(J_0, 2.0/3.0);

		const double I_1 = F * F;
		const double J = get_J(F);
		Assert(J > 0, dealii::ExcMessage("The determinant of the deformation gradient must be greater than zero"));

		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
		}

		dealii::Vector<double> dI_1(9);
		dealii::FullMatrix<double> d2I_1;
		for(unsigned int m = 0; m < 9; ++m)
			dI_1[m] = 2.0 * F[m];
		if(get<2>(requested_quantities))
		{
			d2I_1.reinit(9,9);
			for(unsigned int m = 0; m < 9; ++m)
				d2I_1(m,m) = 2.0;
		}

		// compute value of potential
		if(get<0>(requested_quantities))
			omega = (0.5 * mu_ * (J_0_23 * I_1 - 3.0 - 2.0 * log(J_0 * J)) + 0.5 * lambda_ * log(J * J_0) * log(J * J_0)) / J_0;

		// first derivatives of potential w.r.t. J and I_1
		const double dpsi_dI_1 = (0.5 * mu_ * J_0_23) / J_0;
		const double dpsi_dJ = (-mu_/J + lambda_ * log(J * J_0) / J) / J_0;

		// first derivative
		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = dpsi_dI_1 * dI_1[m] + dpsi_dJ * dJ_dF[m];
		}

		// second derivative
		if(get<2>(requested_quantities))
		{
			const double d2psi_dJ2 = (mu_/J/J + lambda_ * (1.0/J/J) * (1.0 - log(J * J_0))) / J_0;
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < 9; ++n)
					d2_omega(m, n) = dpsi_dI_1 * d2I_1(m, n) + dpsi_dJ * d2J_dF2(m, n) + d2psi_dJ2 * dJ_dF[m] * dJ_dF[n];
		}
		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{

		double factor = 2.0;
		dealii::Vector<double> e(e_omega.size());

		while(true)
		{
			for(unsigned int m = 0; m < e.size(); ++m)
				e[m] = e_omega[m] + factor * delta_e_omega[m];
			if(get_J(e) > 0.0)
				return factor;
			factor *= 0.5;
			Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient stays positive!"));
		}

		return factor;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::modify_K_cell_f_cell
	 */
	void
	modify_K_cell_f_cell(	const dealii::GalerkinTools::DomainCellDoFIterator<spacedim>&	domain_cell,
							dealii::FullMatrix<double>&										K_cell,
							dealii::Vector<double>&											f_cell,
							const dealii::Vector<double>&									/*solution*/,
							const dealii::Vector<double>&									/*solution_C*/,
							const std::vector<unsigned int>&								scalar_functional_indices_to_cell_shapefuns,
							const std::vector<unsigned int>&								/*scalar_functional_indices_to_independent_scalar_indices*/)
	const
	{
		if(ignore_dof_indices != nullptr)
		{
			static vector<unsigned int> dof_indices_local_global;
			dof_indices_local_global.resize(domain_cell->get_fe().n_dofs_per_cell());
			domain_cell.get_dof_indices(dof_indices_local_global);
			for(unsigned int shapefun = 0; shapefun < scalar_functional_indices_to_cell_shapefuns.size(); ++shapefun)
			{
				if(ignore_dof_indices->find(dof_indices_local_global[scalar_functional_indices_to_cell_shapefuns[shapefun]]) != ignore_dof_indices->end())
				{
					for(unsigned int n = 0; n < K_cell.n(); ++n)
						K_cell(shapefun, n) = K_cell(n, shapefun) = 0.0;
					f_cell(shapefun) = 0.0;
				}
			}
		}
	}

};


/**
 *
 * Class defining a scalar functional enforcing the equilibrium equation associated with a compressible Neo-Hooke material by means of an Lagrangian multiplier
 *
 * The integrand is
 *
 * \f$h^\Omega_\rho = \boldsymbol{\lambda} : \left( \mu \boldsymbol{F} + \lambda \dfrac{\ln J}{J} - \mu \dfrac{1}{J} \right)\f$,
 *
 * where
 *
 * \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$, \f$\boldsymbol{\lambda}\f$ are the Lagrangian multipliers,
 * and \f$\mu\f$ and \f$\lambda\f$ Lame's constants.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0]  \f$F_{xx}\f$ <br>
 * 																					[1]  \f$F_{xy}\f$ <br>
 * 																					[2]  \f$F_{xz}\f$ <br>
 * 																					[3]  \f$F_{yx}\f$ <br>
 * 																					[4]  \f$F_{yy}\f$ <br>
 * 																					[5]  \f$F_{yz}\f$ <br>
 * 																					[6]  \f$F_{zx}\f$ <br>
 * 																					[7]  \f$F_{zy}\f$ <br>
 * 																					[8]  \f$F_{zz}\f$ <br>
 * 																					[9]  \f$\lambda_{xx}\f$ <br>
 * 																					[10] \f$\lambda_{xy}\f$ <br>
 * 																					[11] \f$\lambda_{xz}\f$ <br>
 * 																					[12] \f$\lambda_{yx}\f$ <br>
 * 																					[13] \f$\lambda_{yy}\f$ <br>
 * 																					[14] \f$\lambda_{yz}\f$ <br>
 * 																					[15] \f$\lambda_{zx}\f$ <br>
 * 																					[16] \f$\lambda_{zy}\f$ <br>
 * 																					[17] \f$\lambda_{zz}\f$
 */
template<unsigned int spacedim>
class PsiNeoHookeLagrange00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\mu\f$
	 */
	const double
	mu;

	/**
	 * %Function allowing to scale \f$\lambda\f$ and \f$\mu\f$ in dependence on position (the function must provide with the scaling factor)
	 */
	const dealii::Function<spacedim>&
	scaling_function;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		lambda					PsiNeoHookeLagrange00::lambda
	 *
	 * @param[in]		mu						PsiNeoHookeLagrange00::mu
	 *
	 * @param[in]		scaling_function		PsiNeoHookeLagrange00::scaling_function
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiNeoHookeLagrange00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																lambda,
							const double																mu,
							const dealii::Function<spacedim>&											scaling_function,
							const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiNeoHookeLagrange00"),
	lambda(lambda),
	mu(mu),
	scaling_function(scaling_function)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		// scale mu and lambda
		const double mu_ = scaling_function.value(x) * mu;
		const double lambda_ = scaling_function.value(x) * lambda;

		// deformation gradient and derived quantities
		dealii::Vector<double> F(9), L(9);
		for(unsigned int m = 0; m < 9; ++m)
		{
			F[m] = values[m];
			L[m] = values[m + 9];
		}

		const double J = get_J(F);
		Assert(J > 0, dealii::ExcMessage("The determinant of the deformation gradient must be greater than zero"));
		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2(9,9);
		get_dJ_dF(F, dJ_dF);
		get_d2J_dF2(F, d2J_dF2);
		const double dJ_dF_L = dJ_dF * L;
		dealii::Vector<double> d2J_dF2_L(9);
		d2J_dF2.vmult(d2J_dF2_L, L);

		// first derivatives of potential w.r.t. J and I_1
		const double dpsi_dJ = (-mu_ + lambda_ * log(J))/J;
		const double d2psi_dJ2 = (mu_ - lambda_ * log(J) + lambda_)/J/J;

		// compute value of potential
		if(get<0>(requested_quantities))
		{
			omega = 0.0;
			for(unsigned int m = 0; m < 9; ++m)
				omega += L[m] * ( mu_ * F[m] + dpsi_dJ * dJ_dF[m] );
		}


		// first derivative
		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
			{
				d_omega[m] = L[m] * mu_ + dpsi_dJ * d2J_dF2_L[m] + d2psi_dJ2 * dJ_dF_L * dJ_dF[m];
				d_omega[m + 9] = mu_ * F[m] + dpsi_dJ * dJ_dF[m];
			}
		}

		// second derivative
		if(get<2>(requested_quantities))
		{
			dealii::FullMatrix<double> L_d3J_dF3(9,9);
			L_d3J_dF3(0,0) = 0;		L_d3J_dF3(0,1) = 0;		L_d3J_dF3(0,2) = 0;		L_d3J_dF3(0,3) = 0;		L_d3J_dF3(0,4) = L[8];	L_d3J_dF3(0,5) = -L[7];	L_d3J_dF3(0,6) = 0;		L_d3J_dF3(0,7) = -L[5];	L_d3J_dF3(0,8) = L[4];
			L_d3J_dF3(1,0) = 0;		L_d3J_dF3(1,1) = 0;		L_d3J_dF3(1,2) = 0;		L_d3J_dF3(1,3) = -L[8];	L_d3J_dF3(1,4) = 0;		L_d3J_dF3(1,5) = L[6];	L_d3J_dF3(1,6) = L[5];	L_d3J_dF3(1,7) = 0;		L_d3J_dF3(1,8) = -L[3];
			L_d3J_dF3(2,0) = 0;		L_d3J_dF3(2,1) = 0;		L_d3J_dF3(2,2) = 0;		L_d3J_dF3(2,3) = L[7];	L_d3J_dF3(2,4) = -L[6];	L_d3J_dF3(2,5) = 0;		L_d3J_dF3(2,6) = -L[4];	L_d3J_dF3(2,7) = L[3];	L_d3J_dF3(2,8) = 0;
			L_d3J_dF3(3,0) = 0;		L_d3J_dF3(3,1) = -L[8];	L_d3J_dF3(3,2) = L[7];	L_d3J_dF3(3,3) = 0;		L_d3J_dF3(3,4) = 0;		L_d3J_dF3(3,5) = 0;		L_d3J_dF3(3,6) = 0;		L_d3J_dF3(3,7) = L[2];	L_d3J_dF3(3,8) = -L[1];
			L_d3J_dF3(4,0) = L[8];	L_d3J_dF3(4,1) = 0;		L_d3J_dF3(4,2) = -L[6];	L_d3J_dF3(4,3) = 0;		L_d3J_dF3(4,4) = 0;		L_d3J_dF3(4,5) = 0;		L_d3J_dF3(4,6) = -L[2];	L_d3J_dF3(4,7) = 0;		L_d3J_dF3(4,8) = L[0];
			L_d3J_dF3(5,0) = -L[7];	L_d3J_dF3(5,1) = L[6];	L_d3J_dF3(5,2) = 0;		L_d3J_dF3(5,3) = 0;		L_d3J_dF3(5,4) = 0;		L_d3J_dF3(5,5) = 0;		L_d3J_dF3(5,6) = L[1];	L_d3J_dF3(5,7) = -L[0];	L_d3J_dF3(5,8) = 0;
			L_d3J_dF3(6,0) = 0;		L_d3J_dF3(6,1) = L[5];	L_d3J_dF3(6,2) = -L[4];	L_d3J_dF3(6,3) = 0;		L_d3J_dF3(6,4) = -L[2];	L_d3J_dF3(6,5) = L[1];	L_d3J_dF3(6,6) = 0;		L_d3J_dF3(6,7) = 0;		L_d3J_dF3(6,8) = 0;
			L_d3J_dF3(7,0) = -L[5];	L_d3J_dF3(7,1) = 0;		L_d3J_dF3(7,2) = L[3];	L_d3J_dF3(7,3) = L[2];	L_d3J_dF3(7,4) = 0;		L_d3J_dF3(7,5) = -L[0];	L_d3J_dF3(7,6) = 0;		L_d3J_dF3(7,7) = 0;		L_d3J_dF3(7,8) = 0;
			L_d3J_dF3(8,0) = L[4];	L_d3J_dF3(8,1) = -L[3];	L_d3J_dF3(8,2) = 0;		L_d3J_dF3(8,3) = -L[1];	L_d3J_dF3(8,4) = L[0];	L_d3J_dF3(8,5) = 0;		L_d3J_dF3(8,6) = 0;		L_d3J_dF3(8,7) = 0;		L_d3J_dF3(8,8) = 0;

			const double dpsi_dJ = (-mu_ + lambda_ * log(J))/J;
			const double d2psi_dJ2 = (mu_ - lambda_ * log(J) + lambda_)/J/J;

			const double d3psi_dJ3 = (-2.0 * mu_ + 2.0 * lambda_ * log(J) - 3.0 * lambda_)/J/J/J;

			for(unsigned int m = 0; m < 9; ++m)
			{
				for(unsigned int n = 0; n < 9; ++n)
				{
					d2_omega(m, n) = dpsi_dJ * L_d3J_dF3(m, n) + d2psi_dJ2 * (d2J_dF2_L[m] * dJ_dF[n] + d2J_dF2_L[n] * dJ_dF[m]) + d3psi_dJ3 * dJ_dF_L * dJ_dF[m] * dJ_dF[n] + d2psi_dJ2 * dJ_dF_L * d2J_dF2(m, n);
					d2_omega(m + 9, n) = d2_omega(m, n + 9) = dpsi_dJ * d2J_dF2(m, n) + d2psi_dJ2 * dJ_dF[m] * dJ_dF[n];
				}
				d2_omega(m + 9, m) += mu_;
				d2_omega(m, m + 9) += mu_;
			}
		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{

		double factor = 2.0;
		dealii::Vector<double> e(9);

		while(true)
		{
			for(unsigned int m = 0; m < 9; ++m)
				e[m] = e_omega[m] + factor * delta_e_omega[m];
			if(get_J(e) > 0.0)
				return factor;

			factor *= 0.5;
			Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient stays positive!"));
		}

		return factor;
	}

};



/**
 * Class defining chemical potential of charged species moving in fluid
 *
 * \f$h^\Omega_\rho = \mu_0 c + RT c \left( \ln\dfrac{c}{c^\mathrm{f}} - 1 \right)\f$,
 *
 * where \f$\mu_0\f$ is a reference chemical potential, \f$R\f$ the gas constant, \f$T\f$ the temperature,
 * \f$c\f$ the species concentration, and \f$c^\mathrm{f}\f$ the fluid concentration.
 *
 * In order to circumvent numerical problems for low species concentrations, the scalar functional may be regularized according to
 *
 * \f$h^\Omega_\rho = RT c^\mathrm{f} \dfrac{c_0}{c^\mathrm{f}_0} h\left( \dfrac{c c^\mathrm{f}_0}{c^\mathrm{f} c_0} \right)\f$,
 *
 * where
 *
 * \f$ h(x) = \begin{cases}
 *          x [ \ln(x)-1] \quad&\mathrm{if}\quad x>\epsilon\\
 *          \epsilon \{ \ln(\epsilon) [ \ln(x) - \ln(\epsilon) + 1] - 1\} \quad&\mathrm{else},
 *         \end{cases} \f$,
 *
 * with \f$\epsilon \ll 1\f$ being a regularization parameter.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$<br>
 * 																					[1] \f$c^\mathrm{f}\f$
 */
template<unsigned int spacedim>
class PsiChemical02 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$RT\f$
	 */
	const double
	RT;

	/**
	 * \f$\mu_0\f$
	 */
	const double
	mu_0;

	/**
	 * \f$\epsilon\f$
	 */
	const double
	eps;

	/**
	 * \f$c_0 / c^\mathrm{f}_0\f$
	 */
	const double
	c_0_c_f_0;

	/**
	 * \f$\ln(\epsilon)\f$
	 */
	const double
	log_eps;

	/**
	 * \f$\ln(c_0 / c^\mathrm{f}_0)\f$
	 */
	const double
	log_c_0_c_f_0;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		RT						PsiChemical02::RT
	 *
	 * @param[in]		mu_0					PsiChemical02::mu_0
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		eps						PsiChemical02::eps
	 *
	 * @param[in]		c_0_c_f_0				PsiChemical02::c_f_c_f_0
	 */
	PsiChemical02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																mu_0,
					const double																alpha,
					const double																eps = 0.0,
					const double																c_0_c_f_0 = 1.0)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical02"),
	RT(RT),
	mu_0(mu_0),
	eps(eps),
	c_0_c_f_0(c_0_c_f_0),
	log_eps(log(eps)),
	log_c_0_c_f_0(log(c_0_c_f_0))
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double c = values[0];
		const double c_f = values[1];

		if( c <= 0.0 )
		{
			std::cout << "Negative concentration" << std::endl;
			return true;
		}

		if( c_f <= 0.0 )
		{
			std::cout << "Negative concentration" << std::endl;
			return true;
		}


		const double log_c_c_f = log(c/c_f);

		const double c_c_0_c_f_0_c_f = (c / c_f) / c_0_c_f_0;

		if(get<0>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				omega = mu_0 * c + RT * c * (log_c_c_f - 1.0);
			}
			else
			{
				omega = (mu_0 + RT * log_c_0_c_f_0) * c + RT * c_f * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps + 1.0 ) - 1.0 );
			}

		}

		if(get<1>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				d_omega[0] = mu_0 + RT * log_c_c_f;
				d_omega[1] = -RT * c/c_f;
			}
			else
			{
				d_omega[0] = mu_0 + RT * log_c_0_c_f_0 + RT * c_f / c * c_0_c_f_0 * eps * log_eps;
				d_omega[1] = RT * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps ) - 1.0 );
			}
		}

		if(get<2>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				d2_omega(0,0) = RT / c;
				d2_omega(1,1) = RT * c / c_f / c_f;
				d2_omega(0,1) = d2_omega(1,0) = -RT / c_f;
			}
			else
			{
				d2_omega(0,0) = -RT * c_f / c / c * c_0_c_f_0 * eps * log_eps;
				d2_omega(1,1) = -RT * c_0_c_f_0 * eps * log_eps / c_f;
				d2_omega(0,1) = d2_omega(1,0) = RT * c_0_c_f_0 * eps * log_eps / c;
			}

		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{
		const double max_step_1 = - e_omega[0] / delta_e_omega[0];
		const double max_step_2 = - e_omega[1] / delta_e_omega[1];
		if( (isnan(max_step_1) || (max_step_1 < 0.0)) && (isnan(max_step_2) || (max_step_2 < 0.0)) )
			return numeric_limits<double>::max();
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

};

/**
 * Class defining chemical potential according to
 *
 * \f$h^\Omega_\rho = \psi_0 c^\mathrm{f}\f$,
 *
 * where \f$c^\mathrm{f} = \dfrac{J - n_0}{V^\mathrm{f}_\mathrm{m}}\f$, \f$\psi_0\f$, \f$n_0\f$,  \f$V^\mathrm{f}_\mathrm{m}\f$ are material parameters and \f$ J = \det\boldsymbol{F}\f$ is the determinant of the deformation gradient \f$\boldsymbol{F}\f$.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$F_{xx}\f$<br>
 * 																					[1] \f$F_{xy}\f$<br>
 * 																					[2] \f$F_{xz}\f$<br>
 * 																					[3] \f$F_{yx}\f$<br>
 * 																					[4] \f$F_{yy}\f$<br>
 * 																					[5] \f$F_{yz}\f$<br>
 * 																					[6] \f$F_{zx}\f$<br>
 * 																					[7] \f$F_{zy}\f$<br>
 * 																					[8] \f$F_{zz}\f$<br>
 */
template<unsigned int spacedim>
class PsiChemical03 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$\psi_0\f$
	 */
	const double
	psi_0;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

	/**
	 * \f$V_\mathrm{m}^\mathrm{f}\f$
	 */
	const double
	V_m_f;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		psi_0					PsiChemical03::psi_0
	 *
	 * @param[in]		n_0						PsiChemical03::n_0
	 *
	 * @param[in]		V_m_f					PsiChemical03::V_m_f
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiChemical03(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																psi_0,
					const double																n_0,
					const double																V_m_f,
					const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical03"),
	psi_0(psi_0),
	n_0(n_0),
	V_m_f(V_m_f)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m];

	 	// J and derivatives
		const double J = get_J(F);
		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
		}

		if(get<0>(requested_quantities))
		{
			omega = psi_0 * ( J - n_0 ) / V_m_f;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = psi_0 / V_m_f * dJ_dF[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < 9; ++n)
					d2_omega(m, n) = psi_0 / V_m_f * d2J_dF2(m, n);
		}

		return false;
	}
};


/**
 * Class defining chemical potential of charged species moving in fluid
 *
 * \f$h^\Omega_\rho = \mu_0 c + RT c \left( \ln\dfrac{c}{c^\mathrm{f}} - 1 \right)\f$,
 *
 * where \f$\mu_0\f$ is a reference chemical potential, \f$R\f$ the gas constant, \f$T\f$ the temperature,
 * \f$c\f$ the species concentration, \f$c^\mathrm{f} = \dfrac{J - n_0}{V^\mathrm{f}_\mathrm{m}}\f$ the fluid concentration, \f$ J = \det\boldsymbol{F}\f$ the
 * determinant of the deformation gradient \f$\boldsymbol{F}\f$, and \f$n_0\f$ and \f$V^\mathrm{f}_\mathrm{m}\f$ are constants.
 *
 * In order to circumvent numerical problems for low species concentrations, the scalar functional may be regularized according to
 *
 * \f$h^\Omega_\rho = RT c^\mathrm{f} \dfrac{c_0}{c^\mathrm{f}_0} h\left( \dfrac{c c^\mathrm{f}_0}{c^\mathrm{f} c_0} \right)\f$,
 *
 * where
 *
 * \f$ h(x) = \begin{cases}
 *          x [ \ln(x)-1] \quad&\mathrm{if}\quad x>\epsilon\\
 *          \epsilon \{ \ln(\epsilon) [ \ln(x) - \ln(\epsilon) + 1] - 1\} \quad&\mathrm{else},
 *         \end{cases} \f$,
 *
 * with \f$\epsilon \ll 1\f$ being a regularization parameter.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$<br>
 * 																					[1] \f$F_{xx}\f$<br>
 * 																					[2] \f$F_{xy}\f$<br>
 * 																					[3] \f$F_{xz}\f$<br>
 * 																					[4] \f$F_{yx}\f$<br>
 * 																					[5] \f$F_{yy}\f$<br>
 * 																					[6] \f$F_{yz}\f$<br>
 * 																					[7] \f$F_{zx}\f$<br>
 * 																					[8] \f$F_{zy}\f$<br>
 * 																					[9] \f$F_{zz}\f$<br>
 */
template<unsigned int spacedim>
class PsiChemical04 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$RT\f$
	 */
	const double
	RT;

	/**
	 * \f$\mu_0\f$
	 */
	const double
	mu_0;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

	/**
	 * \f$V_\mathrm{m}^\mathrm{f}\f$
	 */
	const double
	V_m_f;

	/**
	 * \f$\epsilon\f$
	 */
	const double
	eps;

	/**
	 * \f$c_0 / c^\mathrm{f}_0\f$
	 */
	const double
	c_0_c_f_0;

	/**
	 * \f$\ln(\epsilon)\f$
	 */
	const double
	log_eps;

	/**
	 * \f$\ln(c_0 / c^\mathrm{f}_0)\f$
	 */
	const double
	log_c_0_c_f_0;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		RT						PsiChemical04::RT
	 *
	 * @param[in]		mu_0					PsiChemical04::mu_0
	 *
	 * @param[in]		n_0						PsiChemical04::n_0
	 *
	 * @param[in]		V_m_f					PsiChemical04::V_m_f
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		eps						PsiChemical04::eps
	 *
	 * @param[in]		c_0_c_f_0				PsiChemical04::c_f_c_f_0
	 */
	PsiChemical04(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																mu_0,
					const double																n_0,
					const double																V_m_f,
					const double																alpha,
					const double																eps = 0.0,
					const double																c_0_c_f_0 = 1.0)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical04"),
	RT(RT),
	mu_0(mu_0),
	n_0(n_0),
	V_m_f(V_m_f),
	eps(eps),
	c_0_c_f_0(c_0_c_f_0),
	log_eps(log(eps)),
	log_c_0_c_f_0(log(c_0_c_f_0))
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double c = values[0];
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m + 1];


	 	// J and derivatives
		const double J = get_J(F);
		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
		}

		const double c_f = (J - n_0) / V_m_f;

		if(c == 0)
			return true;

		const double log_c_c_f = log(c/c_f);

		const double c_c_0_c_f_0_c_f = (c / c_f) / c_0_c_f_0;

		if(get<0>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
				omega = mu_0 * c + RT * c * (log_c_c_f - 1.0);
			else
				omega = (mu_0 + RT * log_c_0_c_f_0) * c + RT * c_f * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps + 1.0 ) - 1.0 );

		}

		if(get<1>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				d_omega[0] = mu_0 + RT * log_c_c_f;
				for(unsigned m = 0; m < 9; ++m)
					d_omega[1 + m] = -RT * c/c_f * dJ_dF[m] / V_m_f;
			}
			else
			{
				d_omega[0] = mu_0 + RT * log_c_0_c_f_0 + RT * c_f / c * c_0_c_f_0 * eps * log_eps;
				for(unsigned m = 0; m < 9; ++m)
					d_omega[1 + m] = RT * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps ) - 1.0 ) * dJ_dF[m] / V_m_f;
			}
		}

		if(get<2>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				d2_omega(0,0) = RT / c;
				for(unsigned m = 0; m < 9; ++m)
					for(unsigned n = 0; n < 9; ++n)
						d2_omega(1 + m,1 + n) = RT * c / c_f / c_f * dJ_dF[m] / V_m_f * dJ_dF[n] / V_m_f - RT * c/c_f * d2J_dF2(m,n) / V_m_f;
				for(unsigned m = 0; m < 9; ++m)
					d2_omega(0,1 + m) = d2_omega(1 + m,0) = -RT / c_f  * dJ_dF[m] / V_m_f;
			}
			else
			{
				d2_omega(0,0) = -RT * c_f / c / c * c_0_c_f_0 * eps * log_eps;
				for(unsigned m = 0; m < 9; ++m)
					for(unsigned n = 0; n < 9; ++n)
						d2_omega(1 + m,1 + n) = -RT * c_0_c_f_0 * eps * log_eps / c_f * dJ_dF[m] / V_m_f * dJ_dF[n] / V_m_f + RT * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps ) - 1.0 ) * d2J_dF2(m,n) / V_m_f;
				for(unsigned m = 0; m < 9; ++m)
					d2_omega(0,1 + m) = d2_omega(1 + m,0) = RT * c_0_c_f_0 * eps * log_eps / c * dJ_dF[m] / V_m_f;
			}

//			dealii::LAPACKFullMatrix<double> d2_omega_lapack(d2_omega.m());
//			d2_omega_lapack = d2_omega;
//			d2_omega_lapack.compute_eigenvalues();
//			for(unsigned int m = 0; m < d2_omega.m(); ++m)
//			{
//				cout << d2_omega_lapack.eigenvalue(m).real() << endl;
/*				if(d2_omega_lapack.eigenvalue(m).real() < 0.0)
				{
					cout << "FAILED" << endl;
					AssertThrow(false, dealii::ExcMessage("FAILED"));
				}*/
//			}
//			cout << endl;
		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{
		double max_step = - e_omega[0] / delta_e_omega[0];
		if(isnan(max_step) || (max_step <= 0.0))
			return numeric_limits<double>::max();
		else
			return max_step;
	}

};


/**
 * Class defining chemical potential of charged species moving in fluid
 *
 * \f$h^\Omega_\rho = \mu_0 c + RT c \left( \ln\dfrac{c}{c^\mathrm{f}} - 1 \right)\f$,
 *
 * where \f$\mu_0\f$ is a reference chemical potential, \f$R\f$ the gas constant, \f$T\f$ the temperature,
 * \f$c\f$ the species concentration, \f$c^\mathrm{f} = \dfrac{1 + \mathrm{tr}{\boldsymbol{\varepsilon}} - n_0}{V^\mathrm{f}_\mathrm{m}}\f$ the fluid concentration, \f$ \boldsymbol{\varepsilon}\f$ the
 * infinitesimal strain tensor, and \f$n_0\f$ and \f$V^\mathrm{f}_\mathrm{m}\f$ are constants.
 *
 * In order to circumvent numerical problems for low species concentrations, the scalar functional may be regularized according to
 *
 * \f$h^\Omega_\rho = RT c^\mathrm{f} \dfrac{c_0}{c^\mathrm{f}_0} h\left( \dfrac{c c^\mathrm{f}_0}{c^\mathrm{f} c_0} \right)\f$,
 *
 * where
 *
 * \f$ h(x) = \begin{cases}
 *          x [ \ln(x)-1] \quad&\mathrm{if}\quad x>\epsilon\\
 *          \epsilon \{ \ln(\epsilon) [ \ln(x) - \ln(\epsilon) + 1] - 1\} \quad&\mathrm{else},
 *         \end{cases} \f$,
 *
 * with \f$\epsilon \ll 1\f$ being a regularization parameter.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c\f$<br>
 * 																					[1] \f$\mathrm{tr}{\boldsymbol{\varepsilon}}\f$
 */
template<unsigned int spacedim>
class PsiChemical05 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$RT\f$
	 */
	const double
	RT;

	/**
	 * \f$\mu_0\f$
	 */
	const double
	mu_0;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

	/**
	 * \f$V_\mathrm{m}^\mathrm{f}\f$
	 */
	const double
	V_m_f;

	/**
	 * \f$\epsilon\f$
	 */
	const double
	eps;

	/**
	 * \f$c_0 / c^\mathrm{f}_0\f$
	 */
	const double
	c_0_c_f_0;

	/**
	 * \f$\ln(\epsilon)\f$
	 */
	const double
	log_eps;

	/**
	 * \f$\ln(c_0 / c^\mathrm{f}_0)\f$
	 */
	const double
	log_c_0_c_f_0;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		RT						PsiChemical05::RT
	 *
	 * @param[in]		mu_0					PsiChemical05::mu_0
	 *
	 * @param[in]		n_0						PsiChemical05::n_0
	 *
	 * @param[in]		V_m_f					PsiChemical05::V_m_f
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		eps						PsiChemical05::eps
	 *
	 * @param[in]		c_0_c_f_0				PsiChemical04::c_f_c_f_0
	 */
	PsiChemical05(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																mu_0,
					const double																n_0,
					const double																V_m_f,
					const double																alpha,
					const double																eps = 0.0,
					const double																c_0_c_f_0 = 1.0)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical05"),
	RT(RT),
	mu_0(mu_0),
	n_0(n_0),
	V_m_f(V_m_f),
	eps(eps),
	c_0_c_f_0(c_0_c_f_0),
	log_eps(log(eps)),
	log_c_0_c_f_0(log(c_0_c_f_0))
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double c = values[0];
		const double c_f = (1 + values[1] - n_0) / V_m_f;

		if(c == 0)
			return true;

		const double log_c_c_f = log(c/c_f);

		const double c_c_0_c_f_0_c_f = (c / c_f) / c_0_c_f_0;

		if(get<0>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				omega = mu_0 * c + RT * c * (log_c_c_f - 1.0);
			}
			else
			{
				omega = (mu_0 + RT * log_c_0_c_f_0) * c + RT * c_f * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps + 1.0 ) - 1.0 );
			}

		}

		if(get<1>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				d_omega[0] = mu_0 + RT * log_c_c_f;
				d_omega[1] = -RT * c/c_f * 1 / V_m_f;
			}
			else
			{
				d_omega[0] = mu_0 + RT * log_c_0_c_f_0 + RT * c_f / c * c_0_c_f_0 * eps * log_eps;
				d_omega[1] = RT * c_0_c_f_0 * eps * ( log_eps * ( log_c_c_f - log_c_0_c_f_0 - log_eps ) - 1.0 ) / V_m_f;
			}
		}

		if(get<2>(requested_quantities))
		{
			if(c_c_0_c_f_0_c_f > eps)
			{
				d2_omega(0,0) = RT / c;
				d2_omega(1,1) = RT * c / c_f / c_f / V_m_f / V_m_f;
				d2_omega(0,1) = d2_omega(1,0) = -RT / c_f / V_m_f;
			}
			else
			{
				d2_omega(0,0) = -RT * c_f / c / c * c_0_c_f_0 * eps * log_eps;
				d2_omega(1,1) = -RT * c_0_c_f_0 * eps * log_eps / c_f / V_m_f / V_m_f;
				d2_omega(0,1) = d2_omega(1,0) = RT * c_0_c_f_0 * eps * log_eps / c / V_m_f;
			}
		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{
		double max_step = - e_omega[0] / delta_e_omega[0];
		if(isnan(max_step) || (max_step <= 0.0))
			return numeric_limits<double>::max();
		else
			return max_step;
	}

};


/**
 * Class defining the free energy of mixing between solvent and polymeric backbone for a hydrogel
 *
 * \f$h^\Omega_\rho = \dfrac{RT}{V^\mathrm{f}_\mathrm{m}} \left[ (J-n_0)\ln\left( 1-\dfrac{n_0}{J} \right) - \chi n^2_0 \dfrac{1}{J}  \right]\f$,
 *
 * where \f$R\f$ is the gas constant, \f$T\f$ the absolute temperature, \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of solvent, \f$n_0\f$ the initial volume fraction
 * of the polymer, \f$\chi\f$ a material parameter, and \f$J = \det\boldsymbol{F}\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$F_{xx}\f$<br>
 * 																					[1] \f$F_{xy}\f$<br>
 * 																					[2] \f$F_{xz}\f$<br>
 * 																					[3] \f$F_{yx}\f$<br>
 * 																					[4] \f$F_{yy}\f$<br>
 * 																					[5] \f$F_{yz}\f$<br>
 * 																					[6] \f$F_{zx}\f$<br>
 * 																					[7] \f$F_{zy}\f$<br>
 * 																					[8] \f$F_{zz}\f$<br>
*/
template<unsigned int spacedim>
class PsiChemical06 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$R T\f$
	 */
	const double
	RT;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

	/**
	 * \f$V^\mathrm{f}_\mathrm{m}\f$
	 */
	const double
	V_m_f;

	/**
	 * \f$\chi\f$
	 */
	const double
	chi;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		RT						PsiChemical06::RT
	 *
	 * @param[in]		n_0						PsiChemical06::n_0
	 *
	 * @param[in]		V_m_f					PsiChemical06::V_m_f
	 *
	 * @param[in]		chi						PsiChemical06::chi
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiChemical06(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																n_0,
					const double																V_m_f,
					const double																chi,
					const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical06"),
	RT(RT),
	n_0(n_0),
	V_m_f(V_m_f),
	chi(chi)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m];

	 	// J and derivatives
		const double J = get_J(F);
		Assert(J > 0, dealii::ExcMessage("The determinant of the deformation gradient must be greater than zero"));
		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		const double domega_dJ = RT / V_m_f * ( log(1 - n_0 / J) + n_0 / J + chi * n_0 * n_0 / J / J );
		double d2omega_dJ2 = 0.0;
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
			d2omega_dJ2 = RT / V_m_f * n_0 / J / J * ( J / (J - n_0) - 1.0 - 2.0 * chi * n_0 / J );
		}

		if(get<0>(requested_quantities))
		{
			omega = RT * (J - n_0) / V_m_f * log(1 - n_0 / J) - RT * chi * n_0 * n_0 / V_m_f / J;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = domega_dJ * dJ_dF[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < 9; ++n)
					d2_omega(m, n) = domega_dJ * d2J_dF2(m,n) + d2omega_dJ2 * dJ_dF[m] * dJ_dF[n];
		}

		return false;
	}

};


/**
 * Class defining the free energy of mixing between solvent and polymeric backbone for a hydrogel
 *
 * \f$h^\Omega_\rho = \dfrac{RT}{V^\mathrm{f}_\mathrm{m}} \left[ (J-n_0)\ln\left( 1-\dfrac{n_0}{J} \right) - \chi n^2_0 \dfrac{1}{J}  \right]\f$,
 *
 * where \f$R\f$ is the gas constant, \f$T\f$ the absolute temperature, \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of solvent, \f$n_0\f$ the initial volume fraction
 * of the polymer, \f$\chi\f$ a material parameter, and \f$J = c^\mathrm{f} V^\mathrm{f}_\mathrm{m} + n_0\f$ the determinant of the deformation gradient expressed in terms of the solvent concentration \f$c^\mathrm{f}\f$.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$c^\mathrm{f}\f$
*/
template<unsigned int spacedim>
class PsiChemical07 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$R T\f$
	 */
	const double
	RT;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

	/**
	 * \f$V^\mathrm{f}_\mathrm{m}\f$
	 */
	const double
	V_m_f;

	/**
	 * \f$\chi\f$
	 */
	const double
	chi;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		RT						PsiChemical07::RT
	 *
	 * @param[in]		n_0						PsiChemical07::n_0
	 *
	 * @param[in]		V_m_f					PsiChemical07::V_m_f
	 *
	 * @param[in]		chi						PsiChemical07::chi
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiChemical07(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																n_0,
					const double																V_m_f,
					const double																chi,
					const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical07"),
	RT(RT),
	n_0(n_0),
	V_m_f(V_m_f),
	chi(chi)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const double J = V_m_f * values[0] + n_0;

		if(get<0>(requested_quantities))
		{
			omega = RT * (J - n_0) / V_m_f * log(1 - n_0 / J) - RT * chi * n_0 * n_0 / V_m_f / J;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = RT * ( log(1 - n_0 / J) + n_0 / J + chi * n_0 * n_0 / J / J );
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0,0) = V_m_f * RT * n_0 / J / J * ( J / (J - n_0) - 1.0 - 2.0 * chi * n_0 / J );
		}

		return false;
	}

};



/**
 * Class defining an incompressibility constraint for a hydrogel through the Lagrangian multiplier term
 *
 * \f$h^\Omega_\rho = p\left( n_0 + c^\mathrm{f} V^\mathrm{f}_\mathrm{m} - J \right)\f$,
 *
 * where \f$n_0\f$ is the initial concentration of the solid skeleton, \f$c^\mathrm{f}\f$ the fluid concentration,
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid, and \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$F_{xx}\f$<br>
 * 																					[1] \f$F_{xy}\f$<br>
 * 																					[2] \f$F_{xz}\f$<br>
 * 																					[3] \f$F_{yx}\f$<br>
 * 																					[4] \f$F_{yy}\f$<br>
 * 																					[5] \f$F_{yz}\f$<br>
 * 																					[6] \f$F_{zx}\f$<br>
 * 																					[7] \f$F_{zy}\f$<br>
 * 																					[8] \f$F_{zz}\f$<br>
 * 																					[9] \f$c^\mathrm{f}\f$<br>
 * 																					[10] \f$p\f$
*/
template<unsigned int spacedim>
class PsiIncompressibility00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$V^\mathrm{f}_\mathrm{m}\f$
	 */
	const double
	V_m_f;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

	/**
	 * If @p true, deformation gradient is considered as parameter in potential (i.e., it is held fixed when computing the first derivative)
	 */
	const bool
	F_as_parameter;

	/**
	 * If @p true, \f$c^\mathrm{f}\f$ is considered as parameter in potential (i.e., it is held fixed when computing the first derivative)
	 */
	const bool
	c_f_as_parameter;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		V_m_f					PsiIncompressibility00::V_m_f
	 *
	 * @param[in]		n_0						PsiIncompressibility00::n_0
	 *
	 * @param[in]		F_as_parameter			PsiIncompressibility00::F_as_parameter
	 *
	 * @param[in]		c_f_as_parameter		PsiIncompressibility00::c_f_as_parameter
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiIncompressibility00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																V_m_f,
							const double																n_0,
							const bool																	F_as_parameter,
							const bool																	c_f_as_parameter,
							const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiIncompressibility00"),
	V_m_f(V_m_f),
	n_0(n_0),
	F_as_parameter(F_as_parameter),
	c_f_as_parameter(c_f_as_parameter)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m];
		const double c_f = values[9];
		const double p = values[10];

	 	// J and derivatives
		const double J = get_J(F);
		Assert(J > 0, dealii::ExcMessage("The determinant of the deformation gradient must be greater than zero"));
		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
		}

		if(get<0>(requested_quantities))
		{
			if(F_as_parameter)
				// potential does anyway not exist in this case
				omega = 0.0;
			else
				omega = p * ( V_m_f * c_f + n_0 - J );
		}

		if(get<1>(requested_quantities))
		{
			d_omega.reinit(11);
			// don't calculate derivatives w.r.t. deformation gradient if it is only a parameter
			if(!F_as_parameter)
			{
				for(unsigned int m = 0; m < 9; ++m)
					d_omega[m] = -p * dJ_dF[m];
			}
			if(!c_f_as_parameter)
				d_omega[9] = p * V_m_f;
			d_omega[10] = V_m_f * c_f + n_0 - J;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega.reinit(11,11);
			for(unsigned int m = 0; m < 9; ++m)
			{
				if(!F_as_parameter)
				{
					for(unsigned int n = 0; n < 9; ++n)
						d2_omega(m, n) = -p * d2J_dF2(m, n);
					d2_omega(m, 10) = -dJ_dF[m];
				}
				d2_omega(10, m) = -dJ_dF[m];

			}
			if(!c_f_as_parameter)
				d2_omega(9, 10) = V_m_f;
			d2_omega(10, 9) = V_m_f;
		}

		return false;
	}

};


/**
 * Class defining an incompressibility constraint in that the determinant of a symmetric or unsymmetric rank 2 tensor is constrained to 1.
 *
 * \f$h^\Omega_\rho = -p\left( J - 1 \right)\f$,
 *
 * where \f$J=\det(\boldsymbol{Q})\f$, \f$\boldsymbol{Q}\f$ is a rank 2 tensor and \f$p\f$ the Lagrangian multiplier (the "pressure").
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega for symmetric case:<br>	[0] \f$Q_{xx}\f$<br>
 * 																									[1] \f$Q_{xy}\f$<br>
 * 																									[2] \f$Q_{xz}\f$<br>
 * 																									[3] \f$Q_{yy}\f$<br>
 * 																									[4] \f$Q_{yz}\f$<br>
 * 																									[5] \f$Q_{zz}\f$<br>
 * 																									[6] \f$p\f$
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega for unsymmetric case:<br>	[0] \f$Q_{xx}\f$<br>
 * 																										[1] \f$Q_{xy}\f$<br>
 * 																										[2] \f$Q_{xz}\f$<br>
 * 																										[3] \f$Q_{xy}\f$<br>
 * 																										[4] \f$Q_{yy}\f$<br>
 * 																										[5] \f$Q_{yz}\f$<br>
 * 																										[6] \f$Q_{zx}\f$<br>
 * 																										[7] \f$Q_{zy}\f$<br>
 * 																										[8] \f$Q_{zz}\f$<br>
 * 																										[9] \f$p\f$

*/
template<unsigned int spacedim>
class PsiIncompressibility01 : public incrementalFE::Psi<spacedim, spacedim>
{
private:

	/**
	 * Indicates whether \f$\boldsymbol{Q}\f$ is symmetric
	 */
	const bool
	symmetric = true;


public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		symmetric				PsiIncompressibility01::symmetric
	 */
	PsiIncompressibility01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																alpha,
							const bool																	symmetric = false)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiIncompressibility01"),
	symmetric(symmetric)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const unsigned int N = symmetric ? 6 : 9;
		dealii::Vector<double> Q(N);
		for(unsigned int m = 0; m < N; ++m)
			Q[m] = values[m];
		const double p = values[N];

	 	// J and derivatives
		const double J = get_J(Q, symmetric);
		Assert(J > 0, dealii::ExcMessage("The determinant of Q must be greater than zero"));
		dealii::Vector<double> dJ_dQ(N);
		dealii::FullMatrix<double> d2J_dQ2;
		get_dJ_dF(Q, dJ_dQ, symmetric);
		if(get<2>(requested_quantities))
		{
			d2J_dQ2.reinit(N,N);
			get_d2J_dF2(Q, d2J_dQ2, symmetric);
		}

		if(get<0>(requested_quantities))
		{
			omega = p * ( 1.0 - J );
		}

		if(get<1>(requested_quantities))
		{
			// don't calculate derivatives w.r.t. deformation gradient if it is only a parameter
			for(unsigned int m = 0; m < N; ++m)
				d_omega[m] = -p * dJ_dQ[m];
			d_omega[N] = 1.0 - J;
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < N; ++m)
			{
				for(unsigned int n = 0; n < N; ++n)
					d2_omega(m, n) = -p * d2J_dQ2(m, n);
				d2_omega(N, m) = d2_omega(m, N) = -dJ_dQ[m];
			}
		}

		return false;
	}

};


/**
 * Class defining an initial pressure term
 *
 * \f$h^\Omega_\rho = p_0 J\f$,
 *
 * where \f$p_0\f$ is the (constant) initial pressure, and \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$F_{xx}\f$<br>
 * 																					[1] \f$F_{xy}\f$<br>
 * 																					[2] \f$F_{xz}\f$<br>
 * 																					[3] \f$F_{yx}\f$<br>
 * 																					[4] \f$F_{yy}\f$<br>
 * 																					[5] \f$F_{yz}\f$<br>
 * 																					[6] \f$F_{zx}\f$<br>
 * 																					[7] \f$F_{zy}\f$<br>
 * 																					[8] \f$F_{zz}\f$<br>
*/
template<unsigned int spacedim>
class PsiInitialPressure00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * \f$p_0\f$
	 */
	const double
	p_0;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		p_0						PsiInitialPressure00::p_0
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiInitialPressure00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																p_0,
							const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiInitialPressure00"),
	p_0(p_0)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		dealii::Vector<double> F(9);
		for(unsigned int m = 0; m < 9; ++m)
			F[m] = values[m];

	 	// J and derivatives
		const double J = get_J(F);
		Assert(J > 0, dealii::ExcMessage("The determinant of the deformation gradient must be greater than zero"));
		dealii::Vector<double> dJ_dF(9);
		dealii::FullMatrix<double> d2J_dF2;
		get_dJ_dF(F, dJ_dF);
		if(get<2>(requested_quantities))
		{
			d2J_dF2.reinit(9,9);
			get_d2J_dF2(F, d2J_dF2);
		}

		if(get<0>(requested_quantities))
		{
			omega = p_0 * J;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = p_0 * dJ_dF[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < 9; ++n)
					d2_omega(m, n) = p_0 * d2J_dF2(m,n);
		}

		return false;
	}

};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = \dfrac{\lambda}{2} \left[\mathrm{tr}\left(\boldsymbol{E}^\mathrm{e}\right)\right]^2 + \mu\mathrm{tr}\left[\left(\boldsymbol{E}^\mathrm{e}\right)^2\right]\f$,
 *
 * where
 *
 * \f$ \boldsymbol{E}^\mathrm{e} = \boldsymbol{Q} \cdot \left(\boldsymbol{E} + \dfrac{1}{2} \boldsymbol{I} \right) \cdot \boldsymbol{Q} - \dfrac{1}{2} \mathbf{I} \f$
 *
 * and
 *
 * \f$ \boldsymbol{E} = \dfrac{1}{2} ( \boldsymbol{F}^\top \cdot \boldsymbol{F} - \boldsymbol{I} ) \f$
 *
 * Here, \f$\boldsymbol{F}\f$ is the deformation gradient, \f$\boldsymbol{Q}\f$ the inverse of the plastic right stretch tensor, and \f$\lambda\f$ and \f$\mu\f$ are Lame's constants.
 * The scalar functional describes
 * the strain energy of an elastic plastic material, with elastic strains assumed to be small and plastic strains large. The ansatz is based on the multiplicative split
 * \f$ \boldsymbol{F} = \boldsymbol{F}^\mathrm{e} \cdot \boldsymbol{Q}^{-1} \f$ of the deformation gradient, where \f$\boldsymbol{F}^\mathrm{e}\f$ is the elastic deformation
 * contribution, which is, without loss of generality, assumed to include a rotation part.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0]   \f$F_{xx}\f$ <br>
 * 																					[1]   \f$F_{xy}\f$ <br>
 * 																					[2]   \f$F_{xz}\f$ <br>
 * 																					[3]   \f$F_{yx}\f$ <br>
 * 																					[4]   \f$F_{yy}\f$ <br>
 * 																					[5]   \f$F_{yz}\f$ <br>
 * 																					[6]   \f$F_{zx}\f$ <br>
 * 																					[7]   \f$F_{zy}\f$ <br>
 * 																					[8]   \f$F_{zz}\f$ <br>
 * 																					[9]   \f$Q_{xx}\f$ <br>
 * 																					[10]  \f$Q_{xy}\f$ <br>
 * 																					[11]  \f$Q_{xz}\f$ <br>
 * 																					[12]  \f$Q_{yy}\f$ <br>
 * 																					[13]  \f$Q_{yz}\f$ <br>
 * 																					[14]  \f$Q_{zz}\f$
 */
template<unsigned int spacedim>
class PsiElasticPlasticMaterial00 : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\mu\f$
	 */
	const double
	mu;

	/**
	 * Maximum allowable plastic strain increment in an iteration
	 */
	const double
	max_strain_increment;


public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		lambda					PsiElasticPlasticMaterial00::lambda
	 *
	 * @param[in]		mu						PsiElasticPlasticMaterial00::mu
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 *
	 * @param[in]		max_strain_increment	PsiElasticPlasticMaterial00::max_strain_increment
	 */
	PsiElasticPlasticMaterial00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																lambda,
								const double																mu,
								const double																alpha,
								const double																max_strain_increment = 0.01)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "ElasticPlasticMaterial00"),
	lambda(lambda),
	mu(mu),
	max_strain_increment(max_strain_increment)
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		// don't waste time with memory allocations
		static dealii::Vector<double> T(6), dPsi_dE(6), dPsi_dQ(6), dPsi_dF(9);
		static dealii::FullMatrix<double> dE_dF(6,9), A(6,6), B(6,6), dT_dE(6,6), dT_dQ(6,6), dA_dQ_T(6,6), dB_dQ_T(6,6), P(9,9), d2Psi_dE2(6,6), d2Psi_dE_dQ(6,6), d2Psi_dQ2(6,6), d2Psi_dF_dE(9,6), d2Psi_dF2(9,9), d2Psi_dF_dQ(9,6);

		// note: expressions have been obtained by using Octave's symbolic package and the ccode code generation feature.
		// the compiler should be able to optimize this by common subexpression replacement

		const double F_11 = values[0];	const double F_12 = values[1];	const double F_13 = values[2];
		const double F_21 = values[3];	const double F_22 = values[4];	const double F_23 = values[5];
		const double F_31 = values[6];	const double F_32 = values[7];	const double F_33 = values[8];

		const double Q_11 = values[9];	const double Q_12 = values[10];	const double Q_13 = values[11];
										const double Q_22 = values[12];	const double Q_23 = values[13];
																		const double Q_33 = values[14];

		const double E_11 = 0.5*(F_11 * F_11) + 0.5*(F_21 * F_21) + 0.5*(F_31 * F_31) - 0.5;	const double E_12 = 0.5*F_11*F_12 + 0.5*F_21*F_22 + 0.5*F_31*F_32;						const double E_13 = 0.5*F_11*F_13 + 0.5*F_21*F_23 + 0.5*F_31*F_33;
																								const double E_22 = 0.5*(F_12 * F_12) + 0.5*(F_22 * F_22) + 0.5*(F_32 * F_32) - 0.5;	const double E_23 = 0.5*F_12*F_13 + 0.5*F_22*F_23 + 0.5*F_32*F_33;
																																														const double E_33 = 0.5*(F_13 * F_13) + 0.5*(F_23* F_23) + 0.5*(F_33* F_33) - 0.5;

		if(get<1>(requested_quantities) || get<2>(requested_quantities))
		{
			dE_dF(0,0) = F_11;			dE_dF(0,1) = 0;			dE_dF(0,2) = 0;			dE_dF(0,3) = F_21;			dE_dF(0,4) = 0;			dE_dF(0,5) = 0;			dE_dF(0,6) = F_31;			dE_dF(0,7) = 0;			dE_dF(0,8) = 0;
			dE_dF(1,0) = 0.5*F_12;		dE_dF(1,1) = 0.5*F_11;	dE_dF(1,2) = 0;			dE_dF(1,3) = 0.5*F_22;		dE_dF(1,4) = 0.5*F_21;	dE_dF(1,5) = 0;			dE_dF(1,6) = 0.5*F_32;		dE_dF(1,7) = 0.5*F_31;	dE_dF(1,8) = 0;
			dE_dF(2,0) = 0.5*F_13;		dE_dF(2,1) = 0;			dE_dF(2,2) = 0.5*F_11;	dE_dF(2,3) = 0.5*F_23;		dE_dF(2,4) = 0;			dE_dF(2,5) = 0.5*F_21;	dE_dF(2,6) = 0.5*F_33;		dE_dF(2,7) = 0;			dE_dF(2,8) = 0.5*F_31;
			dE_dF(3,0) = 0;				dE_dF(3,1) = F_12;		dE_dF(3,2) = 0;			dE_dF(3,3) = 0;				dE_dF(3,4) = F_22;		dE_dF(3,5) = 0;			dE_dF(3,6) = 0;				dE_dF(3,7) = F_32;		dE_dF(3,8) = 0;
			dE_dF(4,0) = 0;				dE_dF(4,1) = 0.5*F_13;	dE_dF(4,2) = 0.5*F_12;	dE_dF(4,3) = 0;				dE_dF(4,4) = 0.5*F_23;	dE_dF(4,5) = 0.5*F_22;	dE_dF(4,6) = 0;				dE_dF(4,7) = 0.5*F_33;	dE_dF(4,8) = 0.5*F_32;
			dE_dF(5,0) = 0;				dE_dF(5,1) = 0;			dE_dF(5,2) = F_13;		dE_dF(5,3) = 0;				dE_dF(5,4) = 0;			dE_dF(5,5) = F_23;		dE_dF(5,6) = 0;				dE_dF(5,7) = 0;			dE_dF(5,8) = F_33;
		}

		if(get<1>(requested_quantities) || get<2>(requested_quantities))
		{

			T[0] = lambda*  (Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 3.0/2.0) + 2.0*mu*(Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) - 0.5);
			T[1] = 2.0*mu*  (Q_12*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_22*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_23*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)));
			T[2] = 2.0*mu*  (Q_13*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_23*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_33*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)));
			T[3] = lambda*  (Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 3.0/2.0) + 2.0*mu*(Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) - 0.5);
			T[4] = 2.0*mu*  (Q_13*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_23*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_33*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)));
			T[5] = lambda*  (Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 3.0/2.0) + 2.0*mu*(Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 0.5);

			A(0,0) = Q_11*Q_11;		A(0,1) = 2.0*Q_11*Q_12;					A(0,2) = 2.0*Q_11*Q_13;					A(0,3) = Q_12*Q_12;		A(0,4) = 2.0*Q_12*Q_13;					A(0,5) = Q_13*Q_13;
			A(1,0) = 2.0*Q_11*Q_12;	A(1,1) = 2.0*Q_11*Q_22 + 2.0*Q_12*Q_12;	A(1,2) = 2.0*Q_11*Q_23 + 2.0*Q_12*Q_13;	A(1,3) = 2.0*Q_12*Q_22;	A(1,4) = 2.0*Q_12*Q_23 + 2.0*Q_13*Q_22;	A(1,5) = 2.0*Q_13*Q_23;
			A(2,0) = 2.0*Q_11*Q_13;	A(2,1) = 2.0*Q_11*Q_23 + 2.0*Q_12*Q_13;	A(2,2) = 2.0*Q_11*Q_33 + 2.0*Q_13*Q_13;	A(2,3) = 2.0*Q_12*Q_23;	A(2,4) = 2.0*Q_12*Q_33 + 2.0*Q_13*Q_23;	A(2,5) = 2.0*Q_13*Q_33;
			A(3,0) = Q_12*Q_12;		A(3,1) = 2.0*Q_12*Q_22;					A(3,2) = 2.0*Q_12*Q_23;					A(3,3) = Q_22*Q_22;		A(3,4) = 2.0*Q_22*Q_23;					A(3,5) = Q_23*Q_23;
			A(4,0) = 2.0*Q_12*Q_13;	A(4,1) = 2.0*Q_12*Q_23 + 2.0*Q_13*Q_22;	A(4,2) = 2.0*Q_12*Q_33 + 2.0*Q_13*Q_23;	A(4,3) = 2.0*Q_22*Q_23;	A(4,4) = 2.0*Q_22*Q_33 + 2.0*Q_23*Q_23;	A(4,5) = 2.0*Q_23*Q_33;
			A(5,0) = Q_13*Q_13;		A(5,1) = 2.0*Q_13*Q_23;					A(5,2) = 2.0*Q_13*Q_33;					A(5,3) = Q_23*Q_23;		A(5,4) = 2.0*Q_23*Q_33;					A(5,5) = Q_33*Q_33;

			B(0,0) = 2.0*E_12*Q_12 + 2.0*E_13*Q_13 + 2.0*Q_11*(E_11 + 0.5);			B(0,1) = 2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*Q_12*(E_11 + 0.5);											B(0,2) = 2.0*E_12*Q_23 + 2.0*E_13*Q_33 + 2.0*Q_13*(E_11 + 0.5);											B(0,3) = 0;															B(0,4) = 0;																								B(0,5) = 0;
			B(1,0) = 2.0*E_12*Q_11 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_22 + 0.5);			B(1,1) = 4.0*E_12*Q_12 + 2.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_11*(E_11 + 0.5) + 2.0*Q_22*(E_22 + 0.5);	B(1,2) = 2.0*E_12*Q_13 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5);											B(1,3) = 2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*Q_12*(E_11 + 0.5);		B(1,4) = 2.0*E_12*Q_23 + 2.0*E_13*Q_33 + 2.0*Q_13*(E_11 + 0.5);											B(1,5) = 0;
			B(2,0) = 2.0*E_13*Q_11 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_33 + 0.5);			B(2,1) = 2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*Q_23*(E_33 + 0.5);											B(2,2) = 2.0*E_12*Q_12 + 4.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_11*(E_11 + 0.5) + 2.0*Q_33*(E_33 + 0.5);	B(2,3) = 0;															B(2,4) = 2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*Q_12*(E_11 + 0.5);											B(2,5) = 2.0*E_12*Q_23 + 2.0*E_13*Q_33 + 2.0*Q_13*(E_11 + 0.5);
			B(3,0) = 0;																B(3,1) = 2.0*E_12*Q_11 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_22 + 0.5);											B(3,2) = 0;																								B(3,3) = 2.0*E_12*Q_12 + 2.0*E_23*Q_23 + 2.0*Q_22*(E_22 + 0.5);		B(3,4) = 2.0*E_12*Q_13 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5);											B(3,5) = 0;
			B(4,0) = 0;																B(4,1) = 2.0*E_13*Q_11 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_33 + 0.5);											B(4,2) = 2.0*E_12*Q_11 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_22 + 0.5);											B(4,3) = 2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*Q_23*(E_33 + 0.5);		B(4,4) = 2.0*E_12*Q_12 + 2.0*E_13*Q_13 + 4.0*E_23*Q_23 + 2.0*Q_22*(E_22 + 0.5) + 2.0*Q_33*(E_33 + 0.5);	B(4,5) = 2.0*E_12*Q_13 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5);
			B(5,0) = 0;																B(5,1) = 0;																								B(5,2) = 2.0*E_13*Q_11 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_33 + 0.5);											B(5,3) = 0;															B(5,4) = 2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*Q_23*(E_33 + 0.5);											B(5,5) = 2.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_33*(E_33 + 0.5);

			A.vmult(dPsi_dE, T);
		}

		if(get<2>(requested_quantities))
		{
			dT_dE(0,0) = 2.0*Q_11*Q_11*mu + lambda*(Q_11*Q_11 + Q_12*Q_12 + Q_13*Q_13);			dT_dE(0,1) = 4.0*Q_11*Q_12*mu + lambda*(2.0*Q_11*Q_12 + 2.0*Q_12*Q_22 + 2.0*Q_13*Q_23);			dT_dE(0,2) = 4.0*Q_11*Q_13*mu + lambda*(2.0*Q_11*Q_13 + 2.0*Q_12*Q_23 + 2.0*Q_13*Q_33);			dT_dE(0,3) = 2.0*Q_12*Q_12*mu + lambda*(Q_12*Q_12 + Q_22*Q_22 + Q_23*Q_23);			dT_dE(0,4) = 4.0*Q_12*Q_13*mu + lambda*(2.0*Q_12*Q_13 + 2.0*Q_22*Q_23 + 2.0*Q_23*Q_33);			dT_dE(0,5) = 2.0*Q_13*Q_13*mu + lambda*(Q_13*Q_13 + Q_23*Q_23 + Q_33*Q_33);
			dT_dE(1,0) = 2.0*Q_11*Q_12*mu;														dT_dE(1,1) = 2.0*mu*(Q_11*Q_22 + Q_12*Q_12);													dT_dE(1,2) = 2.0*mu*(Q_11*Q_23 + Q_12*Q_13);													dT_dE(1,3) = 2.0*Q_12*Q_22*mu;														dT_dE(1,4) = 2.0*mu*(Q_12*Q_23 + Q_13*Q_22);													dT_dE(1,5) = 2.0*Q_13*Q_23*mu;
			dT_dE(2,0) = 2.0*Q_11*Q_13*mu;														dT_dE(2,1) = 2.0*mu*(Q_11*Q_23 + Q_12*Q_13);													dT_dE(2,2) = 2.0*mu*(Q_11*Q_33 + Q_13*Q_13);													dT_dE(2,3) = 2.0*Q_12*Q_23*mu;														dT_dE(2,4) = 2.0*mu*(Q_12*Q_33 + Q_13*Q_23);													dT_dE(2,5) = 2.0*Q_13*Q_33*mu;
			dT_dE(3,0) = 2.0*Q_12*Q_12*mu + lambda*(Q_11*Q_11 + Q_12*Q_12 + Q_13*Q_13);			dT_dE(3,1) = 4.0*Q_12*Q_22*mu + lambda*(2.0*Q_11*Q_12 + 2.0*Q_12*Q_22 + 2.0*Q_13*Q_23);			dT_dE(3,2) = 4.0*Q_12*Q_23*mu + lambda*(2.0*Q_11*Q_13 + 2.0*Q_12*Q_23 + 2.0*Q_13*Q_33);			dT_dE(3,3) = 2.0*Q_22*Q_22*mu + lambda*(Q_12*Q_12 + Q_22*Q_22 + Q_23*Q_23);			dT_dE(3,4) = 4.0*Q_22*Q_23*mu + lambda*(2.0*Q_12*Q_13 + 2.0*Q_22*Q_23 + 2.0*Q_23*Q_33);			dT_dE(3,5) = 2.0*Q_23*Q_23*mu + lambda*(Q_13*Q_13 + Q_23*Q_23 + Q_33*Q_33);
			dT_dE(4,0) = 2.0*Q_12*Q_13*mu;														dT_dE(4,1) = 2.0*mu*(Q_12*Q_23 + Q_13*Q_22);													dT_dE(4,2) = 2.0*mu*(Q_12*Q_33 + Q_13*Q_23);													dT_dE(4,3) = 2.0*Q_22*Q_23*mu;														dT_dE(4,4) = 2.0*mu*(Q_22*Q_33 + Q_23*Q_23);													dT_dE(4,5) = 2.0*Q_23*Q_33*mu;
			dT_dE(5,0) = 2.0*Q_13*Q_13*mu + lambda*(Q_11*Q_11 + Q_12*Q_12 + Q_13*Q_13);			dT_dE(5,1) = 4.0*Q_13*Q_23*mu + lambda*(2.0*Q_11*Q_12 + 2.0*Q_12*Q_22 + 2.0*Q_13*Q_23);			dT_dE(5,2) = 4.0*Q_13*Q_33*mu + lambda*(2.0*Q_11*Q_13 + 2.0*Q_12*Q_23 + 2.0*Q_13*Q_33);			dT_dE(5,3) = 2.0*Q_23*Q_23*mu + lambda*(Q_12*Q_12 + Q_22*Q_22 + Q_23*Q_23);			dT_dE(5,4) = 4.0*Q_23*Q_33*mu + lambda*(2.0*Q_12*Q_13 + 2.0*Q_22*Q_23 + 2.0*Q_23*Q_33);			dT_dE(5,5) = 2.0*Q_33*Q_33*mu + lambda*(Q_13*Q_13 + Q_23*Q_23 + Q_33*Q_33);

			dT_dQ(0,0) = lambda*(2.0*E_12*Q_12 + 2.0*E_13*Q_13 + 2.0*Q_11*(E_11 + 0.5)) + 2.0*mu*(2.0*E_12*Q_12 + 2.0*E_13*Q_13 + 2.0*Q_11*(E_11 + 0.5));	dT_dQ(0,1) = lambda*(2.0*E_12*Q_11 + 2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_11 + 0.5) + 2.0*Q_12*(E_22 + 0.5)) + 2.0*mu*(2.0*E_12*Q_11 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_22 + 0.5));			dT_dQ(0,2) = lambda*(2.0*E_12*Q_23 + 2.0*E_13*Q_11 + 2.0*E_13*Q_33 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_11 + 0.5) + 2.0*Q_13*(E_33 + 0.5)) + 2.0*mu*(2.0*E_13*Q_11 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_33 + 0.5));	dT_dQ(0,3) = lambda*(2.0*E_12*Q_12 + 2.0*E_23*Q_23 + 2.0*Q_22*(E_22 + 0.5));																	dT_dQ(0,4) = lambda*(2.0*E_12*Q_13 + 2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5) + 2.0*Q_23*(E_33 + 0.5));																			dT_dQ(0,5) = lambda*(2.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_33*(E_33 + 0.5));
			dT_dQ(1,0) = 2.0*mu*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5));																				dT_dQ(1,1) = 2.0*mu*(2.0*E_12*Q_12 + E_13*Q_13 + E_23*Q_23 + Q_11*(E_11 + 0.5) + Q_22*(E_22 + 0.5));																											dT_dQ(1,2) = 2.0*mu*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5));																																		dT_dQ(1,3) = 2.0*mu*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5));																				dT_dQ(1,4) = 2.0*mu*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5));																																				dT_dQ(1,5) = 0;
			dT_dQ(2,0) = 2.0*mu*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5));																				dT_dQ(2,1) = 2.0*mu*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5));																																				dT_dQ(2,2) = 2.0*mu*(E_12*Q_12 + 2.0*E_13*Q_13 + E_23*Q_23 + Q_11*(E_11 + 0.5) + Q_33*(E_33 + 0.5));																									dT_dQ(2,3) = 0;																																	dT_dQ(2,4) = 2.0*mu*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5));																																				dT_dQ(2,5) = 2.0*mu*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5));
			dT_dQ(3,0) = lambda*(2.0*E_12*Q_12 + 2.0*E_13*Q_13 + 2.0*Q_11*(E_11 + 0.5));																	dT_dQ(3,1) = lambda*(2.0*E_12*Q_11 + 2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_11 + 0.5) + 2.0*Q_12*(E_22 + 0.5)) + 2.0*mu*(2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*Q_12*(E_11 + 0.5));			dT_dQ(3,2) = lambda*(2.0*E_12*Q_23 + 2.0*E_13*Q_11 + 2.0*E_13*Q_33 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_11 + 0.5) + 2.0*Q_13*(E_33 + 0.5));																	dT_dQ(3,3) = lambda*(2.0*E_12*Q_12 + 2.0*E_23*Q_23 + 2.0*Q_22*(E_22 + 0.5)) + 2.0*mu*(2.0*E_12*Q_12 + 2.0*E_23*Q_23 + 2.0*Q_22*(E_22 + 0.5));	dT_dQ(3,4) = lambda*(2.0*E_12*Q_13 + 2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5) + 2.0*Q_23*(E_33 + 0.5)) + 2.0*mu*(2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*Q_23*(E_33 + 0.5));			dT_dQ(3,5) = lambda*(2.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_33*(E_33 + 0.5));
			dT_dQ(4,0) = 0;																																	dT_dQ(4,1) = 2.0*mu*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5));																																				dT_dQ(4,2) = 2.0*mu*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5));																																		dT_dQ(4,3) = 2.0*mu*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5));																				dT_dQ(4,4) = 2.0*mu*(E_12*Q_12 + E_13*Q_13 + 2.0*E_23*Q_23 + Q_22*(E_22 + 0.5) + Q_33*(E_33 + 0.5));																											dT_dQ(4,5) = 2.0*mu*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5));
			dT_dQ(5,0) = lambda*(2.0*E_12*Q_12 + 2.0*E_13*Q_13 + 2.0*Q_11*(E_11 + 0.5));																	dT_dQ(5,1) = lambda*(2.0*E_12*Q_11 + 2.0*E_12*Q_22 + 2.0*E_13*Q_23 + 2.0*E_23*Q_13 + 2.0*Q_12*(E_11 + 0.5) + 2.0*Q_12*(E_22 + 0.5));																			dT_dQ(5,2) = lambda*(2.0*E_12*Q_23 + 2.0*E_13*Q_11 + 2.0*E_13*Q_33 + 2.0*E_23*Q_12 + 2.0*Q_13*(E_11 + 0.5) + 2.0*Q_13*(E_33 + 0.5)) + 2.0*mu*(2.0*E_12*Q_23 + 2.0*E_13*Q_33 + 2.0*Q_13*(E_11 + 0.5));	dT_dQ(5,3) = lambda*(2.0*E_12*Q_12 + 2.0*E_23*Q_23 + 2.0*Q_22*(E_22 + 0.5));																	dT_dQ(5,4) = lambda*(2.0*E_12*Q_13 + 2.0*E_13*Q_12 + 2.0*E_23*Q_22 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5) + 2.0*Q_23*(E_33 + 0.5)) + 2.0*mu*(2.0*E_12*Q_13 + 2.0*E_23*Q_33 + 2.0*Q_23*(E_22 + 0.5));			dT_dQ(5,5) = lambda*(2.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_33*(E_33 + 0.5)) + 2.0*mu*(2.0*E_13*Q_13 + 2.0*E_23*Q_23 + 2.0*Q_33*(E_33 + 0.5));

			dA_dQ_T(0,0) = 2.0*Q_11*T[0] + 2.0*Q_12*T[1] + 2.0*Q_13*T[2];	dA_dQ_T(0,1) = 2.0*Q_11*T[1] + 2.0*Q_12*T[3] + 2.0*Q_13*T[4];									dA_dQ_T(0,2) = 2.0*Q_11*T[2] + 2.0*Q_12*T[4] + 2.0*Q_13*T[5];									dA_dQ_T(0,3) = 0;												dA_dQ_T(0,4) = 0;																				dA_dQ_T(0,5) = 0;
			dA_dQ_T(1,0) = 2.0*Q_12*T[0] + 2.0*Q_22*T[1] + 2.0*Q_23*T[2];	dA_dQ_T(1,1) = 2.0*Q_11*T[0] + 4.0*Q_12*T[1] + 2.0*Q_13*T[2] + 2.0*Q_22*T[3] + 2.0*Q_23*T[4];	dA_dQ_T(1,2) = 2.0*Q_12*T[2] + 2.0*Q_22*T[4] + 2.0*Q_23*T[5];									dA_dQ_T(1,3) = 2.0*Q_11*T[1] + 2.0*Q_12*T[3] + 2.0*Q_13*T[4];	dA_dQ_T(1,4) = 2.0*Q_11*T[2] + 2.0*Q_12*T[4] + 2.0*Q_13*T[5];									dA_dQ_T(1,5) = 0;
			dA_dQ_T(2,0) = 2.0*Q_13*T[0] + 2.0*Q_23*T[1] + 2.0*Q_33*T[2];	dA_dQ_T(2,1) = 2.0*Q_13*T[1] + 2.0*Q_23*T[3] + 2.0*Q_33*T[4];									dA_dQ_T(2,2) = 2.0*Q_11*T[0] + 2.0*Q_12*T[1] + 4.0*Q_13*T[2] + 2.0*Q_23*T[4] + 2.0*Q_33*T[5];	dA_dQ_T(2,3) = 0;												dA_dQ_T(2,4) = 2.0*Q_11*T[1] + 2.0*Q_12*T[3] + 2.0*Q_13*T[4];									dA_dQ_T(2,5) = 2.0*Q_11*T[2] + 2.0*Q_12*T[4] + 2.0*Q_13*T[5];
			dA_dQ_T(3,0) = 0;												dA_dQ_T(3,1) = 2.0*Q_12*T[0] + 2.0*Q_22*T[1] + 2.0*Q_23*T[2];									dA_dQ_T(3,2) = 0;																				dA_dQ_T(3,3) = 2.0*Q_12*T[1] + 2.0*Q_22*T[3] + 2.0*Q_23*T[4];	dA_dQ_T(3,4) = 2.0*Q_12*T[2] + 2.0*Q_22*T[4] + 2.0*Q_23*T[5];									dA_dQ_T(3,5) = 0;
			dA_dQ_T(4,0) = 0;												dA_dQ_T(4,1) = 2.0*Q_13*T[0] + 2.0*Q_23*T[1] + 2.0*Q_33*T[2];									dA_dQ_T(4,2) = 2.0*Q_12*T[0] + 2.0*Q_22*T[1] + 2.0*Q_23*T[2];									dA_dQ_T(4,3) = 2.0*Q_13*T[1] + 2.0*Q_23*T[3] + 2.0*Q_33*T[4];	dA_dQ_T(4,4) = 2.0*Q_12*T[1] + 2.0*Q_13*T[2] + 2.0*Q_22*T[3] + 4.0*Q_23*T[4] + 2.0*Q_33*T[5];	dA_dQ_T(4,5) = 2.0*Q_12*T[2] + 2.0*Q_22*T[4] + 2.0*Q_23*T[5];
			dA_dQ_T(5,0) = 0;												dA_dQ_T(5,1) = 0;																				dA_dQ_T(5,2) = 2.0*Q_13*T[0] + 2.0*Q_23*T[1] + 2.0*Q_33*T[2];									dA_dQ_T(5,3) = 0;												dA_dQ_T(5,4) = 2.0*Q_13*T[1] + 2.0*Q_23*T[3] + 2.0*Q_33*T[4];									dA_dQ_T(5,5) = 2.0*Q_13*T[2] + 2.0*Q_23*T[4] + 2.0*Q_33*T[5];

			dB_dQ_T(0,0) = 2.0*T[0]*(E_11 + 0.5);					dB_dQ_T(0,1) = 2.0*E_12*T[0] + 2.0*T[1]*(E_11 + 0.5);									dB_dQ_T(0,2) = 2.0*E_13*T[0] + 2.0*T[2]*(E_11 + 0.5);									dB_dQ_T(0,3) = 2.0*E_12*T[1];							dB_dQ_T(0,4) = 2.0*E_12*T[2] + 2.0*E_13*T[1];											dB_dQ_T(0,5) = 2.0*E_13*T[2];
			dB_dQ_T(1,0) = 2.0*E_12*T[0] + 2.0*T[1]*(E_11 + 0.5);	dB_dQ_T(1,1) = 4*E_12*T[1] + 2.0*T[0]*(E_22 + 0.5) + 2.0*T[3]*(E_11 + 0.5);				dB_dQ_T(1,2) = 2.0*E_12*T[2] + 2.0*E_13*T[1] + 2.0*E_23*T[0] + 2.0*T[4]*(E_11 + 0.5);	dB_dQ_T(1,3) = 2.0*E_12*T[3] + 2.0*T[1]*(E_22 + 0.5);	dB_dQ_T(1,4) = 2.0*E_12*T[4] + 2.0*E_13*T[3] + 2.0*E_23*T[1] + 2.0*T[2]*(E_22 + 0.5);	dB_dQ_T(1,5) = 2.0*E_13*T[4] + 2.0*E_23*T[2];
			dB_dQ_T(2,0) = 2.0*E_13*T[0] + 2.0*T[2]*(E_11 + 0.5);	dB_dQ_T(2,1) = 2.0*E_12*T[2] + 2.0*E_13*T[1] + 2.0*E_23*T[0] + 2.0*T[4]*(E_11 + 0.5);	dB_dQ_T(2,2) = 4*E_13*T[2] + 2.0*T[0]*(E_33 + 0.5) + 2.0*T[5]*(E_11 + 0.5);				dB_dQ_T(2,3) = 2.0*E_12*T[4] + 2.0*E_23*T[1];			dB_dQ_T(2,4) = 2.0*E_12*T[5] + 2.0*E_13*T[4] + 2.0*E_23*T[2] + 2.0*T[1]*(E_33 + 0.5);	dB_dQ_T(2,5) = 2.0*E_13*T[5] + 2.0*T[2]*(E_33 + 0.5);
			dB_dQ_T(3,0) = 2.0*E_12*T[1];							dB_dQ_T(3,1) = 2.0*E_12*T[3] + 2.0*T[1]*(E_22 + 0.5);									dB_dQ_T(3,2) = 2.0*E_12*T[4] + 2.0*E_23*T[1];											dB_dQ_T(3,3) = 2.0*T[3]*(E_22 + 0.5);					dB_dQ_T(3,4) = 2.0*E_23*T[3] + 2.0*T[4]*(E_22 + 0.5);									dB_dQ_T(3,5) = 2.0*E_23*T[4];
			dB_dQ_T(4,0) = 2.0*E_12*T[2] + 2.0*E_13*T[1];			dB_dQ_T(4,1) = 2.0*E_12*T[4] + 2.0*E_13*T[3] + 2.0*E_23*T[1] + 2.0*T[2]*(E_22 + 0.5);	dB_dQ_T(4,2) = 2.0*E_12*T[5] + 2.0*E_13*T[4] + 2.0*E_23*T[2] + 2.0*T[1]*(E_33 + 0.5);	dB_dQ_T(4,3) = 2.0*E_23*T[3] + 2.0*T[4]*(E_22 + 0.5);	dB_dQ_T(4,4) = 4*E_23*T[4] + 2.0*T[3]*(E_33 + 0.5) + 2.0*T[5]*(E_22 + 0.5);				dB_dQ_T(4,5) = 2.0*E_23*T[5] + 2.0*T[4]*(E_33 + 0.5);
			dB_dQ_T(5,0) = 2.0*E_13*T[2];							dB_dQ_T(5,1) = 2.0*E_13*T[4] + 2.0*E_23*T[2];											dB_dQ_T(5,2) = 2.0*E_13*T[5] + 2.0*T[2]*(E_33 + 0.5);									dB_dQ_T(5,3) = 2.0*E_23*T[4];							dB_dQ_T(5,4) = 2.0*E_23*T[5] + 2.0*T[4]*(E_33 + 0.5);									dB_dQ_T(5,5) = 2.0*T[5]*(E_33 + 0.5);

			P(0,0) = dPsi_dE[0];		P(0,1) = 0.5*dPsi_dE[1];		P(0,2) = 0.5*dPsi_dE[2];
			P(1,0) = 0.5*dPsi_dE[1];	P(1,1) = dPsi_dE[3];			P(1,2) = 0.5*dPsi_dE[4];
			P(2,0) = 0.5*dPsi_dE[2];	P(2,1) = 0.5*dPsi_dE[4];		P(2,2) = dPsi_dE[5];
			P(3,3) = dPsi_dE[0];		P(3,4) = 0.5*dPsi_dE[1];		P(3,5) = 0.5*dPsi_dE[2];
			P(4,3) = 0.5*dPsi_dE[1];	P(4,4) = dPsi_dE[3];			P(4,5) = 0.5*dPsi_dE[4];
			P(5,3) = 0.5*dPsi_dE[2];	P(5,4) = 0.5*dPsi_dE[4];		P(5,5) = dPsi_dE[5];
			P(6,6) = dPsi_dE[0];		P(6,7) = 0.5*dPsi_dE[1];		P(6,8) = 0.5*dPsi_dE[2];
			P(7,6) = 0.5*dPsi_dE[1];	P(7,7) = dPsi_dE[3];			P(7,8) = 0.5*dPsi_dE[4];
			P(8,6) = 0.5*dPsi_dE[2];	P(8,7) = 0.5*dPsi_dE[4];		P(8,8) = dPsi_dE[5];
		}

		if(get<0>(requested_quantities))
		{
			const double tr_E_e = Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5))+ Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 1.5;
			omega = 0.5 * lambda * tr_E_e * tr_E_e
					+ mu*(	2.0*(Q_11*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_12*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_13*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)))
							   *(Q_12*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_22*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_23*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)))
						  + 2.0*(Q_11*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_12*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_13*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)))
						       *(Q_13*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_23*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_33*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)))
						  + 2.0*(Q_12*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_22*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_23*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)))
						       *(Q_13*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_23*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_33*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)))
						  + 1.0*(Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) - 0.5)
						       * (Q_11*(E_12*Q_12 + E_13*Q_13 + Q_11*(E_11 + 0.5)) + Q_12*(E_12*Q_11 + E_23*Q_13 + Q_12*(E_22 + 0.5)) + Q_13*(E_13*Q_11 + E_23*Q_12 + Q_13*(E_33 + 0.5)) - 0.5)
						  + 1.0*(Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) - 0.5)
						       * (Q_12*(E_12*Q_22 + E_13*Q_23 + Q_12*(E_11 + 0.5)) + Q_22*(E_12*Q_12 + E_23*Q_23 + Q_22*(E_22 + 0.5)) + Q_23*(E_13*Q_12 + E_23*Q_22 + Q_23*(E_33 + 0.5)) - 0.5)
						  + 1.0*(Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 0.5)
						       * (Q_13*(E_12*Q_23 + E_13*Q_33 + Q_13*(E_11 + 0.5)) + Q_23*(E_12*Q_13 + E_23*Q_33 + Q_23*(E_22 + 0.5)) + Q_33*(E_13*Q_13 + E_23*Q_23 + Q_33*(E_33 + 0.5)) - 0.5));
		}

		if(get<1>(requested_quantities))
		{
			B.vmult(dPsi_dQ, T);
			dE_dF.Tvmult(dPsi_dF, dPsi_dE);
			for(unsigned int i = 0; i < 9; ++i)
				d_omega[i] = dPsi_dF[i];
			for(unsigned int i = 9; i < 15; ++i)
				d_omega[i] = dPsi_dQ[i - 9];
		}

		if(get<2>(requested_quantities))
		{
			d2Psi_dF2 = P;
			d2Psi_dE_dQ = dA_dQ_T;
			d2Psi_dQ2 = dB_dQ_T;

			A.mmult(d2Psi_dE2, dT_dE);
			A.mmult(d2Psi_dE_dQ, dT_dQ, true);
			B.mmult(d2Psi_dQ2, dT_dQ, true);
			dE_dF.Tmmult(d2Psi_dF_dE, d2Psi_dE2);
			dE_dF.TmTmult(d2Psi_dF2, d2Psi_dF_dE, true);
			dE_dF.Tmmult(d2Psi_dF_dQ, d2Psi_dE_dQ);
			for(unsigned int i = 0; i < 9; ++i)
				for(unsigned int j = 0; j < 9; ++j)
					d2_omega(i,j) = d2Psi_dF2(i,j);
			for(unsigned int i = 0; i < 9; ++i)
				for(unsigned int j = 9; j < 15; ++j)
					d2_omega(i,j) = d2_omega(j,i) = d2Psi_dF_dQ(i,j - 9);
			for(unsigned int i = 9; i < 15; ++i)
				for(unsigned int j = 9; j < 15; ++j)
					d2_omega(i,j) = d2Psi_dQ2(i - 9,j - 9);
		}

		return false;
	}

	/**
	 * see ScalarFunctional<spacedim, spacedim>::get_maximum_step
	 */
	double
	get_maximum_step(	const dealii::Vector<double>& 				e_omega,
						const std::vector<dealii::Vector<double>>&	/*e_omega_ref_sets*/,
						const dealii::Vector<double>& 				delta_e_omega,
						const dealii::Vector<double>& 				/*hidden_vars*/,
						const dealii::Point<spacedim>& 				/*x*/)
	const
	{

		double factor = 2.0;
		static dealii::Vector<double> F(9), Q(6);
		while(true)
		{

			for(unsigned int m = 0; m < 9; ++m)
				F[m] = e_omega[m] + factor * delta_e_omega[m];
			for(unsigned int m = 9; m < 15; ++m)
				Q[m - 9] = e_omega[m] + factor * delta_e_omega[m];
			if( (get_J(F) > 0.0) && (get_J(Q, true) > 0.0))
				break;

			factor *= 0.5;
			Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient and that of Q stays positive!"));
		}
		return factor;
	}

};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = P_{\varphi\varphi}(R F_{\varphi\varphi}-u_R)\f$,
 *
 * where \f$P_{\varphi\varphi}\f$ is a Lagrange multiplier, \f$u_R\f$ the radial displacement, \f$R\f$ the radial coordinate, and \f$\varepsilon_{\varphi\varphi}\f$ the circumferential strain.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$P_{\varphi\varphi}\f$<br>
 * 																					[1] \f$\varepsilon_{\varphi\varphi}\f$<br>
 * 																					[2] \f$u_R\f$
 */
template<unsigned int spacedim>
class PsiCylindricalSymmetry00 : public incrementalFE::Psi<spacedim, spacedim>
{

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_omega					ScalarFunctional<spacedim, spacedim>::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional<spacedim, spacedim>::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional<spacedim, spacedim>::quadrature
	 *
	 * @param[in]		global_data				Psi<spacedim, spacedim>::global_data
	 *
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiCylindricalSymmetry00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiCylindricalSymmetry00")
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		const double R = x[0];
		Assert(R >= 0, dealii::ExcMessage("Locations with negative radial position are not admissible for cylindrical symmetry"));
		const double P = values[0];
		const double eps = values[1];
		const double u = values[2];

		if(get<0>(requested_quantities))
		{
			omega = P * (R*eps - u);
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = (R*eps - u);
			d_omega[1] = P*R;
			d_omega[2] = -P;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0,0) = 0.0;
			d2_omega(0,1) = R;
			d2_omega(0,2) = -1.0;

			d2_omega(1,0) = R;
			d2_omega(1,1) = 0.0;
			d2_omega(1,2) = 0.0;

			d2_omega(2,0) = -1.0;
			d2_omega(2,1) = 0.0;
			d2_omega(2,2) = 0.0;
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$h^\Sigma_\tau = P_{\varphi\varphi}(R \varepsilon_{\varphi\varphi}-u_R)\f$,
 *
 * where \f$P_{\varphi\varphi}\f$ is a Lagrange multiplier, \f$u_R\f$ the radial displacement, \f$R\f$ the radial coordinate, and \f$\varepsilon_{\varphi\varphi}\f$ the circumferential strain.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>	[0] \f$P_{\varphi\varphi}\f$<br>
 * 																					[1] \f$\varepsilon_{\varphi\varphi}\f$<br>
 * 																					[2] \f$u_R\f$
 */
template<unsigned int spacedim>
class PsiCylindricalSymmetryInterface00 : public incrementalFE::Psi<spacedim-1, spacedim>
{

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_omega
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Psi::global_data
	 *
	 * @param[in]		alpha					Psi::alpha
	 */
	PsiCylindricalSymmetryInterface00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
										const std::set<dealii::types::material_id>										domain_of_integration,
										const dealii::Quadrature<spacedim-1>											quadrature,
										GlobalDataIncrementalFE<spacedim>&												global_data,
										const double																	alpha)
	:
	Psi<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, alpha, "PsiCylindricalSymmetryInterface00")
	{
	}

	/**
	 * @see Psi<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{

		const double R = x[0];
		Assert(R >= 0, dealii::ExcMessage("Locations with negative radial position are not admissible for cylindrical symmetry"));
		const double P = values[0];
		const double eps = values[1];
		const double u = values[2];

		if(get<0>(requested_quantities))
		{
			sigma = P * (R*eps - u);
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = (R*eps - u);
			d_sigma[1] = P*R;
			d_sigma[2] = -P;
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma(0,0) = 0.0;
			d2_sigma(0,1) = R;
			d2_sigma(0,2) = -1.0;

			d2_sigma(1,0) = R;
			d2_sigma(1,1) = 0.0;
			d2_sigma(1,2) = 0.0;

			d2_sigma(2,0) = -1.0;
			d2_sigma(2,1) = 0.0;
			d2_sigma(2,2) = 0.0;
		}

		return false;
	}

};




}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_ */
