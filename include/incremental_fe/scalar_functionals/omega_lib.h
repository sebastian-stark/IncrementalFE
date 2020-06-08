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

#ifndef INCREMENTALFE_SCALARFUNCTIONALS_OMEGALIB_H_
#define INCREMENTALFE_SCALARFUNCTIONALS_OMEGALIB_H_

#include <incremental_fe/scalar_functionals/omega.h>
#include <incremental_fe/fe_model.h>
#include <deal.II/base/exceptions.h>

namespace incrementalFE
{

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ h^\Omega_\rho =	\dfrac{1}{2 D c} \dot{\boldsymbol{I}} \cdot \dot{\boldsymbol{I}} \f$,
 *
 * where \f$c\f$ is the species concentration and \f$\dot{\boldsymbol{I}}\f$ the corresponding flux.
 *
 * The "mobility" \f$D\f$ is related to the usual "diffusion constant" \f$ \bar D \f$ by \f$D = \bar D/(RT)\f$.
 * Moreover, it is related to the electrical mobility \f$\mu\f$ by \f$ D = \mu n/F\f$, where \f$n\f$ is the charge per
 * molecule of the mobile species in multiples of the elementary charge and \f$F\f$ Faraday's constant.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$I_x\f$<br>
 * 																				[1] \f$I_y\f$<br>
 * 																				[2] \f$I_z\f$<br>
 * 																				[3] \f$c\f$
 */
template<unsigned int spacedim>
class OmegaFluxDissipation00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * mobility
	 */
	const double D;

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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		D						OmegaFluxDissipation00::D
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluxDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>									domain_of_integration,
							const dealii::Quadrature<spacedim>											quadrature,
							GlobalDataIncrementalFE<spacedim>&											global_data,
							const double																D,
							const unsigned int															method,
							const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 3, 0, 0, 1, method, alpha, "OmegaFluxDissipation00"),
	D(D)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{
		dealii::Tensor<1, 3> i;
		for(unsigned int m = 0; m < 3; ++m)
			i[m] = values[m];
		const double q = values[3] > 1e-16 ? values[3] : 1e-16;

		if(get<0>(requested_quantities))
		{
			omega = 1.0 / ( 2.0 * D * q) * i * i;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				d_omega[m] = 1.0 / ( D * q ) * i[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				d2_omega(m, m) = 1.0 / ( D * q );

			if(compute_dq && (q > 1e-16))
			{
				for(unsigned int m = 0; m < 3; ++m)
					d2_omega(m, 3) = - 1.0 / ( D * q * q ) * i[m];
			}
		}

		return false;
	}

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ h^\Omega_\rho =	-\dfrac{D c}{2} \boldsymbol{E} \cdot \boldsymbol{E} \f$,
 *
 * where \f$c\f$ is the species concentration and \f$\boldsymbol{E}\f$ the driving force vector for
 * the corresponding species flux.
 *
 * The "mobility" \f$D\f$ is related to the usual "diffusion constant" \f$ \bar D \f$ by \f$D = \bar D/(RT)\f$.
 * Moreover, it is related to the electrical mobility \f$\mu\f$ by \f$ D = \mu n/F\f$, where \f$n\f$ is the charge per
 * molecule of the mobile species in multiples of the elementary charge and \f$F\f$ Faraday's constant.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$E_x\f$<br>
 * 																				[1] \f$E_y\f$<br>
 * 																				[2] \f$E_z\f$<br>
 * 																				[3] \f$c\f$
 */
template<unsigned int spacedim>
class OmegaDualFluxDissipation00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * mobility
	 */
	const double D;

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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		D						OmegaDualFluxDissipation00::D
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualFluxDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																D,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 0, 0, 3, 1, method, alpha, "OmegaDualFluxDissipation00"),
	D(D)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{
		dealii::Tensor<1, 3> E;
		for(unsigned int m = 0; m < 3; ++m)
			E[m] = values[m];
		const double q = values[3];

		if(get<0>(requested_quantities))
		{
			omega = -D * q /  2.0 * E * E;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				d_omega[m] = -D * q * E[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				d2_omega(m, m) = -D * q;

			if(compute_dq)
			{
				for(unsigned int m = 0; m < 3; ++m)
					d2_omega(m, 3) = -D * E[m];
			}
		}


		return false;
	}

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ h^\Omega_\rho = -\dot{c}\eta  \f$,
 *
 * where \f$c\f$ is a species concentration and \f$\eta\f$ the corresponding potential.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$c\f$<br>
 * 																				[1] \f$\eta\f$
 */
template<unsigned int spacedim>
class OmegaMixedTerm00 : public incrementalFE::Omega<spacedim, spacedim>
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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaMixedTerm00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const unsigned int															method,
						const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 0, 1, 1, 0, method, alpha, "OmegaMixedTerm00")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{
		const double c_dot = values[0];
		const double eta = values[1];

		if(get<0>(requested_quantities))
		{
			omega = -c_dot * eta;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = -eta;
			d_omega[1] = -c_dot;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0, 0) = d2_omega(1, 1) = 0.0;
			d2_omega(0, 1) = d2_omega(1, 0) = -1.0;
		}

		return false;
	}

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ h^\Omega_\rho = -\dot{\boldsymbol{D}} \cdot \boldsymbol{E}  \f$,
 *
 * where \f$\boldsymbol{D}\f$ and \f$\boldsymbol{E}\f$ are conjugate fields.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$D_x\f$<br>
 * 																				[1] \f$D_y\f$<br>
 * 																				[2] \f$D_z\f$<br>
 * 																				[3] \f$E_x\f$<br>
 * 																				[4] \f$E_y\f$<br>
 * 																				[5] \f$E_z\f$<br>
 */
template<unsigned int spacedim>
class OmegaMixedTerm01 : public incrementalFE::Omega<spacedim, spacedim>
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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaMixedTerm01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
						const std::set<dealii::types::material_id>									domain_of_integration,
						const dealii::Quadrature<spacedim>											quadrature,
						GlobalDataIncrementalFE<spacedim>&											global_data,
						const unsigned int															method,
						const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 0, 3, 3, 0, method, alpha, "OmegaMixedTerm01")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{
		const double D_x = values[0];
		const double D_y = values[1];
		const double D_z = values[2];
		const double E_x = values[3];
		const double E_y = values[4];
		const double E_z = values[5];

		if(get<0>(requested_quantities))
		{
			omega = - D_x * E_x - D_y * E_y - D_z * E_z;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = -E_x;
			d_omega[1] = -E_y;
			d_omega[2] = -E_z;
			d_omega[3] = -D_x;
			d_omega[4] = -D_y;
			d_omega[5] = -D_z;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0, 3) = d2_omega(3, 0) = -1.0;
			d2_omega(1, 4) = d2_omega(4, 1) = -1.0;
			d2_omega(2, 5) = d2_omega(5, 2) = -1.0;
		}

		return false;
	}

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ h^\Omega_\rho =	-\mu ( \nabla \cdot \dot{\boldsymbol{I}} + \dot{c} ) \f$
 *
 * where \f$\mu\f$ is a Lagrangian multiplier, \f$c\f$ is the species concentration and \f$\dot{\boldsymbol{I}}\f$ the corresponding flux.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0] \f$\nabla \boldsymbol{I}\f$<br>
 * 																				[1] \f$c\f$<br>
 * 																				[2] \f$\mu\f$
 */
template<unsigned int spacedim>
class OmegaDivergenceConstraint00 : public incrementalFE::Omega<spacedim, spacedim>
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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDivergenceConstraint00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 1, 1, 0, method, alpha, "OmegaDivergenceConstraint00")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{
		const double div_I_dot = values[0];
		const double c_dot = values[1];
		const double mu = values[2];


		if(get<0>(requested_quantities))
		{
			omega = -mu * (div_I_dot + c_dot);
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = -mu;
			d_omega[1] = -mu;
			d_omega[2] = -(div_I_dot + c_dot);
		}

		if(get<2>(requested_quantities))
		{
			d2_omega(0, 2) = d2_omega(2, 0) = -1.0;
			d2_omega(1, 2) = d2_omega(2, 1) = -1.0;
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ h^\Sigma_\tau =	-\mu \dot{\boldsymbol{I}} \cdot \boldsymbol{n} \f$
 *
 * where \f$\mu\f$ is a Lagrangian multiplier, and \f$\dot{\boldsymbol{I}}\f$ a flux.
 *
 * This is meant to enforce a zero normal flux condition (e.g. for Raviart-Thomas finite elements).
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$I_x\f$<br>
 * 															[1]	\f$I_y\f$<br>
 * 															[2]	\f$I_z\f$<br>
 * 															[3]	\f$\mu\f$<br>
 */
template<unsigned int spacedim>
class OmegaZeroNormalFlux00 : public incrementalFE::Omega<spacedim-1, spacedim>
{

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaZeroNormalFlux00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const unsigned int																method,
							const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 3, 0, 1, 0, method, alpha, "OmegaZeroNormalFlux00")
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1, spacedim>&	n,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{
		const double I_x_dot = values[0];
		const double I_y_dot = values[1];
		const double I_z_dot = values[2];
		const double mu = values[3];
		const double n_x = n[0];
		const double n_y = n[1];
		const double n_z = spacedim==3 ? n[2] : 0.0;

		if(get<0>(requested_quantities))
		{
			sigma = -mu * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z);
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = -mu * n_x;
			d_sigma[1] = -mu * n_y;
			d_sigma[2] = -mu * n_z;
			d_sigma[3] = -(I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z);
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma(0, 3) = d2_sigma(3, 0) = -n_x;
			d2_sigma(1, 3) = d2_sigma(3, 1) = -n_y;
			d2_sigma(2, 3) = d2_sigma(3, 2) = -n_z;
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ h^\Sigma_\tau =	\bar\mu(t) \dot{\boldsymbol{I}} \cdot \boldsymbol{n} \f$
 *
 * where \f$\bar\mu\f$ is a prescribed potential, and \f$\dot{\boldsymbol{I}}\f$ a flux.
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$I_x\f$<br>
 * 															[1]	\f$I_y\f$<br>
 * 															[2]	\f$I_z\f$<br>
 */
template<unsigned int spacedim>
class OmegaFluxPower00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * %Function determining \f$\bar \mu(t)\f$
	 */
	dealii::Function<spacedim>&
	function_mu;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		function_mu				OmegaFluxPower00::function_mu
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaFluxPower00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						dealii::Function<spacedim>&														function_mu,
						const unsigned int																method,
						const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 3, 0, 0, 0, method, alpha, "OmegaZeroNormalFlux00"),
	function_mu(function_mu)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1, spacedim>&	n,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			/*d2_sigma*/,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{
		const double time_old = function_mu.get_time();
		function_mu.set_time(t);
		const double mu_bar = function_mu.value(x);
		function_mu.set_time(time_old);

		const double I_x_dot = values[0];
		const double I_y_dot = values[1];
		const double I_z_dot = values[2];
		const double n_x = n[0];
		const double n_y = n[1];
		const double n_z = spacedim==3 ? n[2] : 0.0;

		if(get<0>(requested_quantities))
		{
			sigma = mu_bar * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z);
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = mu_bar * n_x;
			d_sigma[1] = mu_bar * n_y;
			d_sigma[2] = mu_bar * n_z;
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ h^\Sigma_\tau =	1/(2D) \left(\dot{\boldsymbol{I}} \cdot \boldsymbol{n}\right)^2 \f$
 *
 * where \f$D\f$ is a dissipation constant, and \f$\dot{\boldsymbol{I}}\f$ a flux.
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$I_x\f$<br>
 * 															[1]	\f$I_y\f$<br>
 * 															[2]	\f$I_z\f$<br>
 */
template<unsigned int spacedim>
class OmegaInterfaceFluxDissipation00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * constant \f$D\f$
	 */
	const double
	D;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		D						OmegaInterfaceFluxDissipation00::D
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaInterfaceFluxDissipation00(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
									const std::set<dealii::types::material_id>										domain_of_integration,
									const dealii::Quadrature<spacedim-1>											quadrature,
									GlobalDataIncrementalFE<spacedim>&												global_data,
									const double																	D,
									const unsigned int																method,
									const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 3, 0, 0, 0, method, alpha, "OmegaInterfaceFluxDissipation00"),
	D(D)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1, spacedim>&	n,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{

		const double I_x_dot = values[0];
		const double I_y_dot = values[1];
		const double I_z_dot = values[2];
		const double n_x = n[0];
		const double n_y = n[1];
		const double n_z = spacedim==3 ? n[2] : 0.0;

		if(get<0>(requested_quantities))
		{
			sigma = 1.0/(2.0*D) * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z) * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z);
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = 1.0/D * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z) * n_x;
			d_sigma[1] = 1.0/D * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z) * n_y;
			d_sigma[2] = 1.0/D * (I_x_dot * n_x + I_y_dot * n_y + I_z_dot * n_z) * n_z;
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma(0,0) = 1.0/D * n_x * n_x;
			d2_sigma(0,1) = 1.0/D * n_y * n_x;
			d2_sigma(0,2) = 1.0/D * n_z * n_x;
			d2_sigma(1,0) = 1.0/D * n_x * n_y;
			d2_sigma(1,1) = 1.0/D * n_y * n_y;
			d2_sigma(1,2) = 1.0/D * n_z * n_y;
			d2_sigma(2,0) = 1.0/D * n_x * n_z;
			d2_sigma(2,1) = 1.0/D * n_y * n_z;
			d2_sigma(2,2) = 1.0/D * n_z * n_z;
		}

		return false;
	}

};

/**
 * Class defining an interface related Butler-Volmer type scalar functional with the integrand
 *
 * \f$ h^\Sigma_\tau =	-I_0\left[ \dfrac{1}{1-\beta} \exp\left( - \dfrac{1-\beta}{RT} \Delta \eta \right) + \dfrac{1}{\beta} \exp\left( \dfrac{\beta}{RT} \Delta \eta \right) \right] \f$
 *
 * where \f$I_0\f$ is related to the exchange current density \f$i_0\f$ by \f$I_0 = i_0 \cdot RT / F \f$, \f$\beta\f$ is the symmetry factor, and \f$\Delta \eta\f$ the thermodynamic driving force.
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\Delta \eta\f$
 */
template<unsigned int spacedim>
class OmegaDualButlerVolmer00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * parameter \f$I_0\f$
	 */
	const double
	I_0;

	/**
	 * parameter \f$\beta\f$
	 */
	const double
	beta;

	/**
	 * \f$RT\f$
	 */
	const double
	RT;

	/**
	 * Threshold parameter \f$\Delta \eta^\mathrm{th}/RT\f$.
	 * If \f$|\Delta \eta/RT| > \Delta \eta^\mathrm{th}/RT\f$, the potential is continued by a quadratic function in order to avoid numerical issues related to large values of the exponential function.
	 */
	const double
	threshold;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		I_0						OmegaDualButlerVolmer00::I_0
	 *
	 * @param[in]		beta					OmegaDualButlerVolmer00::beta
	 *
	 * @param[in]		RT						OmegaDualButlerVolmer00::RT
	 *
	 * @param[in]		threshold				OmegaDualButlerVolmer00::threshold
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaDualButlerVolmer00(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const double																	I_0,
							const double																	beta,
							const double																	RT,
							const double																	threshold,
							const unsigned int																method,
							const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 0, 0, 1, 0, method, alpha, "OmegaDualButlerVolmer00"),
	I_0(I_0),
	beta(beta),
	RT(RT),
	threshold(threshold)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{

		const double eta = values[0];

		double t1, t2;

		if(eta / RT > threshold)
		{
			t1 = exp(beta * threshold);
			t2 = exp( (beta - 1.0) * threshold);
		}
		else if(eta / RT < -threshold)
		{
			t1 = exp(-beta * threshold);
			t2 = exp( -(beta - 1.0) * threshold);
		}
		else
		{
			t1 = exp(beta * eta / RT);
			t2 = exp( (beta - 1.0) * eta / RT);
		}

		if(get<0>(requested_quantities))
		{
			if(eta / RT > threshold)
				sigma = t1 / beta + t2 / (1.0 - beta) + (t1 - t2) * (eta / RT - threshold) + 0.5 * (beta * t1 + (1.0 - beta) * t2) * (eta / RT - threshold) * (eta / RT - threshold);
			else if(eta / RT < -threshold)
				sigma = t1 / beta + t2 / (1.0 - beta) + (t1 - t2) * (eta / RT + threshold) + 0.5 * (beta * t1 + (1.0 - beta) * t2) * (eta / RT + threshold) * (eta / RT + threshold);
			else
				sigma = t1/beta + t2/(1.0 - beta);
			sigma *= -I_0;
		}

		if(get<1>(requested_quantities))
		{
			if(eta / RT > threshold)
			{
				d_sigma[0] = (t1 - t2) + (beta * t1 + (1.0 - beta) * t2) * (eta / RT - threshold);
			}
			else if(eta / RT < -threshold)
			{
				d_sigma[0] = (t1 - t2) + (beta * t1 + (1.0 - beta) * t2) * (eta / RT + threshold);
			}
			else
			{
				d_sigma[0] = (t1 - t2);
			}
			d_sigma[0] *= -I_0 / RT;
		}

		if(get<2>(requested_quantities))
		{
			if(eta / RT > threshold)
			{
				d2_sigma(0,0) = beta * t1 + (1.0 - beta) * t2;
			}
			else if(eta / RT < -threshold)
			{
				d2_sigma(0,0) = beta * t1 + (1.0 - beta) * t2;
			}
			else
			{
				d2_sigma(0,0) = beta * t1 + (1.0 - beta) * t2;
			}
			d2_sigma(0,0) *= -I_0 / RT / RT;
		}

		return false;
	}

};


/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ h^\Sigma_\tau =	\boldsymbol{t} \cdot \left( \boldsymbol{u}^0 - \boldsymbol{u}^1 \right) \f$
 *
 * where \f$\boldsymbol{t}\f$ is a Lagrangian multiplier, \f$\boldsymbol{u}^0\f$ the displacement on the 0 side of the interface, and \f$\boldsymbol{u}^1\f$ the displacement on the 1 side of the interface.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$u^0_x\f$<br>
 * 															[1]	\f$u^0_y\f$<br>
 * 															[2]	\f$u^0_z\f$<br>
 * 															[3] \f$u^1_x\f$<br>
 * 															[4]	\f$u^1_y\f$<br>
 * 															[5]	\f$u^1_z\f$<br>
 * 															[6] \f$t_x\f$<br>
 * 															[7]	\f$t_y\f$<br>
 * 															[8]	\f$t_z\f$<br>
 */
template<unsigned int spacedim>
class OmegaDisplacementCoupling00 : public incrementalFE::Omega<spacedim-1, spacedim>
{

private:

	/**
	 * If this is @p true, \f$\boldsymbol{u}^1\f$ is considered as a parameter (i.e., the corresponding first derivatives are set to zero)
	 */
	const bool
	u_1_as_parameter;

public:

	/**
	 * Constructor
	 *
	 * @param[in]		e_sigma					ScalarFunctional::e_sigma
	 *
	 * @param[in] 		domain_of_integration	ScalarFunctional::domain_of_integration
	 *
	 * @param[in]		quadrature				ScalarFunctional::quadrature
	 *
	 * @param[in]		global_data				Omega::global_data
	 *
	 * @param[in]		u_1_as_parameter		OmegaDisplacementCoupling00::u_1_as_parameter
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaDisplacementCoupling00(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
								const std::set<dealii::types::material_id>										domain_of_integration,
								const dealii::Quadrature<spacedim-1>											quadrature,
								GlobalDataIncrementalFE<spacedim>&												global_data,
								const bool																		u_1_as_parameter,
								const unsigned int																method,
								const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 0, 6, 3, 0, method, alpha, "OmegaDisplacementCoupling00"),
	u_1_as_parameter(u_1_as_parameter)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{

		dealii::Tensor<1,3> u_0, u_1, t;

		for(unsigned int m = 0; m < 3; ++m)
		{
			u_0[m] = values[m];
			u_1[m] = values[m+3];
			t[m] = values[m+6];
		}

		const dealii::Tensor<1,3> u_0_u_1 = u_0 - u_1;


		if(get<0>(requested_quantities))
		{
			sigma = t * u_0_u_1;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_sigma[m] = t[m];
				if(u_1_as_parameter)
					d_sigma[m+3] = 0.0;
				else
					d_sigma[m+3] = -t[m];
				d_sigma[m+6] = u_0_u_1[m];
			}
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma.reinit(9,9);
			for(unsigned int m = 0; m < 3; ++m)
			{
				d2_sigma(m, m+6) = d2_sigma(m+6, m) = 1.0;
				if(!u_1_as_parameter)
					d2_sigma(m+3, m+6) = -1.0;
				d2_sigma(m+6, m+3) = -1.0;
			}
		}

		return false;
	}

};


/**
 * Class defining dual dissipation associated with flux of ions through fluid, which itself possibly flows through solid skeleton.
 *
 * \f$ h^\Omega_\rho =	-\dfrac{D c J V^\mathrm{f}_\mathrm{m}}{2 c^\mathrm{f}} \nabla\eta \cdot \boldsymbol{C}^{-1} \cdot \nabla\eta - \eta \dot{c} \f$,
 *
 * where \f$c\f$ is the species concentration,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * \f$\eta\f$ the electrochemical potential corresponding to \f$c\f$,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$D\f$ the mobility,<br>
 * \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$, <br>
 * and \f$\boldsymbol{C} = \boldsymbol{F}^\top\cdot \boldsymbol{F}\f$.
 *
 * The "mobility" \f$D\f$ is related to the usual "diffusion constant" \f$ \bar D \f$ by \f$D = \bar D/(RT)\f$.
 * Moreover, it is related to the electrical mobility \f$\mu\f$ by \f$ D = \mu n/F\f$, where \f$n\f$ is the charge per
 * molecule of the mobile species in multiples of the elementary charge and \f$F\f$ Faraday's constant.
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{c}\f$<br>
 * 																				[1]  \f$\eta_{,x}\f$<br>
 * 																				[2]  \f$\eta_{,y}\f$<br>
 * 																				[3]  \f$\eta_{,z}\f$<br>
 * 																				[4]  \f$\eta\f$<br>
 * 																				[5]  \f$c\f$<br>
 * 																				[6]  \f$c^\mathrm{f}\f$<br>
 * 																				[7]  \f$F_{xx}\f$<br>
 * 																				[8]  \f$F_{xy}\f$<br>
 * 																				[9]  \f$F_{xz}\f$<br>
 * 																				[10] \f$F_{yx}\f$<br>
 * 																				[11] \f$F_{yy}\f$<br>
 * 																				[12] \f$F_{yz}\f$<br>
 * 																				[13] \f$F_{zx}\f$<br>
 * 																				[14] \f$F_{zy}\f$<br>
 * 																				[15] \f$F_{zz}\f$<br>
 */
template<unsigned int spacedim>
class OmegaDualIonDissipation00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * mobility \f$D\f$
	 */
	const double
	D;

	/**
	 * molar volume of fluid \f$V^\mathrm{f}_\mathrm{m}\f$
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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		D						OmegaDualIonDissipation00::D
	 *
	 * @param[in]		V_m_f					OmegaDualIonDissipation00::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualIonDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																D,
								const double																V_m_f,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 0, 4, 11, method, alpha, "OmegaDualIonDissipation00"),
	D(D),
	V_m_f(V_m_f)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{

		(void)compute_dq;
		Assert(!compute_dq, dealii::ExcMessage("The alpha-family for temporal discretization is not currently implemented!"));

		const double c_dot = values[0];
		dealii::Tensor<1,3> grad_eta;
		for(unsigned int m = 0; m < 3; ++m)
			grad_eta[m] = values[m+1];
		const double eta = values[4];
		const double c = values[5];
		const double c_f = values[6];
		dealii::Tensor<2, 3> F, C, C_inv;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[7 + m * 3 + n];
		C = transpose(F) * F;
		C_inv = invert(C);
		const double n_f = c_f / determinant(F) * V_m_f;
		const dealii::Tensor<1, 3> C_inv_grad_eta = C_inv * grad_eta;
		const double grad_eta_C_inv_grad_eta = C_inv_grad_eta * grad_eta;

		if(get<0>(requested_quantities))
		{
			omega = -0.5 * D * c / n_f * grad_eta_C_inv_grad_eta - eta * c_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = -eta;
			for(unsigned int m = 0; m < 3; ++m)
				d_omega[m + 1] = -D * c / n_f * C_inv_grad_eta[m];
			d_omega[4] = -c_dot;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega.reinit(5,5);
			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int n = 0; n < 3; ++n)
					d2_omega(m + 1, n + 1) = -D * c / n_f * C_inv[m][n];
			d2_omega(0, 4) = d2_omega(4, 0) = -1.0;
		}

		return false;

	}

};


/**
 * Class defining dual dissipation associated with flux of fluid through a solid skeleton, where ions flow in addition through the fluid.
 *
 * \f$ h^\Omega_\rho =	-\dfrac{D J}{2 V^\mathrm{f}_\mathrm{m}}  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) \cdot \boldsymbol{C}^{-1} \cdot \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$c^\mathrm{f}\f$ is the fluid concentration,<br>
 * \f$c^i\f$ are the ion concentrations (\f$i\f$ runs from \f$1\f$ to \f$I\f$),<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential,<br>
 * \f$\eta^i\f$ are the ion potentials,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$J\f$ the determinant of the deformation gradient \f$\boldsymbol{F}\f$,<br>
 * \f$\boldsymbol{C} = \boldsymbol{F}^\top\cdot \boldsymbol{F}\f$,<br>
 * and \f$D\f$ a "fluid mobility".
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  				\f$\dot{c}^\mathrm{f}\f$<br>
 * 																				[1]  				\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[2]  				\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[3]  				\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[4] ... [3+3I]		\f$\eta^i_{,x}\f$, \f$\eta^i_{,y}\f$, \f$\eta^i_{,z}\f$ (ordering: xyz, xyz, ...)<br>
 * 																				[4+3I]				\f$\eta^\mathrm{f}\f$<br>
 * 																				[5+3I]				\f$c^\mathrm{f}\f$<br>
 * 																				[6+3I] ... [5+4I]	\f$c^i\f$<br>
 * 																				[6+4I] ... [14+4I]	\f$F_{xx}\f$, \f$F_{xy}\f$, \f$F_{xz}\f$, \f$F_{yx}\f$, \f$F_{yy}\f$, \f$F_{yz}\f$, \f$F_{zx}\f$, \f$F_{zy}\f$, \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaDualFluidDissipation00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * Number of ionic species \f$I\f$
	 */
	const unsigned int
	I;

	/**
	 * mobility \f$D\f$
	 */
	const double
	D;

	/**
	 * molar volume of fluid \f$V^\mathrm{f}_\mathrm{m}\f$
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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		I						OmegaDualFluidDissipation00::I
	 *
	 * @param[in]		D						OmegaDualFluidDissipation00::D
	 *
	 * @param[in]		V_m_f					OmegaDualFluidDissipation00::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualFluidDissipation00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															I,
								const double																D,
								const double																V_m_f,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 0, 4+3*I, 10+I, method, alpha, "OmegaDualFluidDissipation00"),
	I(I),
	D(D),
	V_m_f(V_m_f)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{

		(void)compute_dq;
		Assert(!compute_dq, dealii::ExcMessage("The alpha-family for temporal discretization is not currently implemented!"));

		// start indices for respective quantities
		const unsigned int i_c_f_dot = 0;
		const unsigned int i_grad_eta_f = 1;
		vector<unsigned int> i_grad_eta_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_grad_eta_i[i] = 4 + 3*i;
		const unsigned int i_eta_f = 4 + 3*I;
		const unsigned int i_c_f = 5 + 3*I;
		vector<unsigned int> i_c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_c_i[i] = 6 + 3*I + i;
		const unsigned int i_F = 6 + 4*I;


		const double c_f_dot = values[i_c_f_dot];
		const double c_f = values[i_c_f];
		dealii::Vector<double> c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			c_i[i] = values[i_c_i[i]];

		dealii::Tensor<1,3> grad_eta;
		for(unsigned int m = 0; m < 3; ++m)
			grad_eta[m] = values[i_grad_eta_f + m];
		for(unsigned int i = 0; i < I; ++i)
			for(unsigned int m = 0; m < 3; ++m)
				grad_eta[m] += c_i[i]/c_f * values[i_grad_eta_i[i] + m];

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2, 3> F, C, C_inv;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		C = transpose(F) * F;
		C_inv = invert(C);
		const dealii::Tensor<1, 3> C_inv_grad_eta = C_inv * grad_eta;
		const double grad_eta_C_inv_grad_eta = C_inv_grad_eta * grad_eta;

		const double  K = D * determinant(F) / V_m_f;

		if(get<0>(requested_quantities))
		{
			omega = -0.5 * K * grad_eta_C_inv_grad_eta - eta_f * c_f_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_c_f_dot] = -eta_f;
			d_omega[i_eta_f]   = -c_f_dot;
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_omega[i_grad_eta_f + m] = -K * C_inv_grad_eta[m];
				for(unsigned int i = 0; i < I; ++i)
					d_omega[i_grad_eta_i[i] + m] = d_omega[i_grad_eta_f + m] * c_i[i]/c_f;
			}
		}

		if(get<2>(requested_quantities))
		{
			d2_omega.reinit(5 + 3*I, 5 + 3*I);

			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int m = 0; m < 3; ++m)
			{
				for(unsigned int n = 0; n < 3; ++n)
				{
					d2_omega[i_grad_eta_f + m][i_grad_eta_f + n] = -K * C_inv[m][n];
					for(unsigned int i = 0; i < I; ++i)
					{
						d2_omega[i_grad_eta_f + m][i_grad_eta_i[i] + n] = d2_omega[i_grad_eta_i[i] + n][i_grad_eta_f + m] = -K * C_inv[m][n] * c_i[i]/c_f;
						for(unsigned int j = 0; j < I; ++j)
							d2_omega[i_grad_eta_i[i] + n][i_grad_eta_i[j] + m] = -K * C_inv[m][n] * (c_i[i] / c_f) * (c_i[j] / c_f);
					}
				}
			}
		}

		return false;

	}

};


/**
 * Class defining Lagrangian multiplier term for equilibrium condition in the case that fluid flows without dissipation, where ions flow in addition with dissipation through the fluid.
 *
 * \f$ h^\Omega_\rho =	\nabla \dot{\xi} \cdot  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$\xi\f$ is a scalar potential,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * \f$c^i\f$ are the ion concentrations (\f$i\f$ runs from \f$1\f$ to \f$I\f$),<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential,<br>
 * and \f$\eta^i\f$ are the ion potentials
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  				\f$\dot{\xi}_{,x}\f$<br>
 * 																				[1]					\f$\dot{\xi}_{,y}\f$<br>
 * 																				[2]					\f$\dot{\xi}_{,z}\f$<br>
 *																				[3]  				\f$\dot{c}^\mathrm{f}\f$<br>
 * 																				[4]  				\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[5]  				\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[6]  				\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[7] ... [6+3I]		\f$\eta^i_{,x}\f$, \f$\eta^i_{,y}\f$, \f$\eta^i_{,z}\f$ (ordering: xyz, xyz, ...)<br>
 * 																				[7+3I]				\f$\eta^\mathrm{f}\f$<br>
 * 																				[8+3I]				\f$c^\mathrm{f}\f$<br>
 * 																				[9+3I] ... [8+4I]	\f$c^i\f$<br>
 */
template<unsigned int spacedim>
class OmegaDualFluidDissipation01 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * Number of ionic species \f$I\f$
	 */
	const unsigned int
	I;

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
	 * @param[in]		global_data				Omega<spacedim, spacedim>::global_data
	 *
	 * @param[in]		I						OmegaDualFluidDissipation01::I
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualFluidDissipation01(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															I,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 3, 4+3*I, 1+I, method, alpha, "OmegaDualFluidDissipation00"),
	I(I)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{

		(void)compute_dq;
		Assert(!compute_dq, dealii::ExcMessage("The alpha-family for temporal discretization is not currently implemented!"));

		// start indices for respective quantities
		const unsigned int i_grad_xi_dot = 0;
		const unsigned int i_c_f_dot = 3;
		const unsigned int i_grad_eta_f = 4;
		vector<unsigned int> i_grad_eta_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_grad_eta_i[i] = 7 + 3*i;
		const unsigned int i_eta_f = 7 + 3*I;
		const unsigned int i_c_f = 8 + 3*I;
		vector<unsigned int> i_c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_c_i[i] = 9 + 3*I + i;

		dealii::Tensor<1,3> grad_xi_dot;
		for(unsigned int m = 0; m < 3; ++m)
			grad_xi_dot[m] = values[i_grad_xi_dot + m];

		const double c_f_dot = values[i_c_f_dot];
		const double c_f = values[i_c_f];
		dealii::Vector<double> c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			c_i[i] = values[i_c_i[i]];

		dealii::Tensor<1,3> grad_eta;
		for(unsigned int m = 0; m < 3; ++m)
			grad_eta[m] = values[i_grad_eta_f + m];
		for(unsigned int i = 0; i < I; ++i)
			for(unsigned int m = 0; m < 3; ++m)
				grad_eta[m] += c_i[i]/c_f * values[i_grad_eta_i[i] + m];

		const double eta_f = values[i_eta_f];


		if(get<0>(requested_quantities))
		{
			omega = grad_xi_dot * grad_eta - eta_f * c_f_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_c_f_dot] = -eta_f;
			d_omega[i_eta_f]   = -c_f_dot;
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_omega[i_grad_xi_dot + m] = grad_eta[m];
				d_omega[i_grad_eta_f + m] = grad_xi_dot[m];
				for(unsigned int i = 0; i < I; ++i)
					d_omega[i_grad_eta_i[i] + m] = grad_xi_dot[m] * c_i[i]/c_f;
			}
		}

		if(get<2>(requested_quantities))
		{
			d2_omega.reinit(8 + 3*I, 8 + 3*I);

			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int m = 0; m < 3; ++m)
			{
				d2_omega[i_grad_xi_dot + m][i_grad_eta_f + m] = d2_omega[i_grad_eta_f + m][i_grad_xi_dot + m] = 1.0;
				for(unsigned int i = 0; i < I; ++i)
					d2_omega[i_grad_xi_dot + m][i_grad_eta_i[i] + m] = d2_omega[i_grad_eta_i[i] + m][i_grad_xi_dot + m] = c_i[i]/c_f;
			}
		}

		return false;

	}
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_OMEGALIB_H_ */
