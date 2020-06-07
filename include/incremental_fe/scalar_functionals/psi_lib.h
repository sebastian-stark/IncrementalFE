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
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>

namespace incrementalFE
{

namespace
{
	double get_J(const dealii::Vector<double>& F)
	{
		return	  F[0] * F[4] * F[8]
				+ F[1] * F[5] * F[6]
				+ F[2] * F[3] * F[7]
				- F[2] * F[4] * F[6]
				- F[1] * F[3] * F[8]
				- F[0] * F[5] * F[7];
	}

	void get_dJ_dF(	const dealii::Vector<double>& 	F,
					dealii::Vector<double>& 		dJ_dF)
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

	void get_d2J_dF2(	const dealii::Vector<double>& F,
						dealii::FullMatrix<double>& d2J_dF2)
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

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = RT c_0 h\left( \dfrac{c}{c_0} \right)\f$,
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
 * for the potential, \f$c\f$ the species concentration, and \f$\epsilon \ll 1\f$ a regularization parameter
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
	 */
	PsiChemical00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																c_0,
					const double																mu_0,
					const double																alpha,
					const double																eps)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical00"),
	RT(RT),
	c_0(c_0),
	mu_0(mu_0),
	eps(eps),
	log_eps(log(eps))
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
		const double log_c_c_0 = log(c/c_0);
		const double c_th = eps*c_0;

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
				d_omega[0] = RT * eps * (c_0/c) * log_eps + mu_0;
			else
				d_omega[0] = RT * log_c_c_0 + mu_0;
		}

		if(get<2>(requested_quantities))
		{
			if(c/c_0 < eps)
				d2_omega(0,0) = - RT * eps * (c_0/c/c) * log_eps +  RT / c_th;
			else
				d2_omega(0,0) = RT / c;
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
			return DBL_MAX;
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
		Assert(J > 0, ExcMessage("The determinant of the deformation gradient must be greater than zero"));

		double tr_C = F * F;

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
			if(get_J(e) > 0.0)
				return factor;

			factor *= 0.5;
			Assert(factor > 0.0, ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient stays positive!"));
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
 * \f$c\f$ the species concentration, and \f$c^\mathrm{f}\f$ the fluid concentration
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
	 */
	PsiChemical02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const double																RT,
					const double																mu_0,
					const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiChemical02"),
	RT(RT),
	mu_0(mu_0)
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

		const double log_c_c_f = log(c/c_f);

		if(get<0>(requested_quantities))
		{
			omega = mu_0 * c + RT * c * (log_c_c_f - 1.0);
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = mu_0 + RT * log_c_c_f;
			d_omega[1] = -RT * c/c_f;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0,0) = RT / c;
			d2_omega(1,1) = RT * c / c_f / c_f;
			d2_omega(0,1) = d2_omega(1,0) = -RT / c_f;
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
	 * @param[in]		alpha					Psi<spacedim, spacedim>::alpha
	 */
	PsiIncompressibility00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																V_m_f,
							const double																n_0,
							const bool																	F_as_parameter,
							const double																alpha)
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, "PsiIncompressibility00"),
	V_m_f(V_m_f),
	n_0(n_0),
	F_as_parameter(F_as_parameter)
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
		Assert(J > 0, ExcMessage("The determinant of the deformation gradient must be greater than zero"));
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
			d2_omega(9, 10) = d2_omega(10, 9) = V_m_f;
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
		Assert(J > 0, ExcMessage("The determinant of the deformation gradient must be greater than zero"));
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



}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_ */
