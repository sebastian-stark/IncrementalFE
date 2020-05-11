#ifndef INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_

#include <incremental_fe/scalar_functionals/psi.h>
#include <incremental_fe/fe_model.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>

namespace incrementalFE
{

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = RTc \left( \ln\dfrac{c}{c_0} - 1 + \mu_0 / (RT) \right)\f$,
 *
 * where \f$R\f$ is the gas constant, \f$T\f$ the temperature, \f$c_0\f$ a
 * reference species concentration, \f$\mu_0\f$ a corresponding reference value
 * for the potential, and \f$c\f$ the species concentration.
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
	 * @param[in]		alpha					ScalarFunctional<spacedim, spacedim>::alpha
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

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$h^\Omega_\rho = \dfrac{a}{2} (c-b)^2\f$,
 *
 * where \f$a\f$ and $b$ are material parameters, and \f$c\f$ the species concentration.
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
	 * @param[in]		alpha					ScalarFunctional<spacedim, spacedim>::alpha
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
 * \f$\dfrac{\mathrm{d} h^\Omega_\rho}{\mathrm{d} p} = p \dfrac{\mathrm{d}c}{\mathrm{d}c}(p) \f$,
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
	 * Function \f$c(p)\f$
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
	 * @param[in]		alpha					ScalarFunctional<spacedim, spacedim>::alpha
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
	 * Vector \f$b\f$
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
	 * @param[in]		alpha					ScalarFunctional<spacedim, spacedim>::alpha
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
	 * Lame's constant \f$\lambda\f$
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
	 * 	 *
	 * @param[in]		alpha					ScalarFunctional<spacedim, spacedim>::alpha
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


}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_PSILIB_H_ */
