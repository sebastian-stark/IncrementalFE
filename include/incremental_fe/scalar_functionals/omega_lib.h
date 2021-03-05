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
#include <incremental_fe/scalar_functionals/psi_lib.h>
#include <incremental_fe/fe_model.h>
#include <deal.II/base/exceptions.h>

namespace incrementalFE
{

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	\dfrac{1}{2 D c} \dot{\boldsymbol{I}} \cdot \dot{\boldsymbol{I}} \f$,
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
 * \f$ \omega^\Omega =	-\dfrac{D c}{2} \boldsymbol{E} \cdot \boldsymbol{E} \f$,
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
 * \f$ \omega^\Omega = -\dot{c}\eta  \f$,
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
 * \f$ \omega^\Omega = -\dot{\boldsymbol{D}} \cdot \boldsymbol{E}  \f$,
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
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	-\dot{I} \eta \f$
 *
 * where \f$\eta\f$ is a potential, and \f$\dot{I}\f$ a flux.
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$I\f$<br>
 * 															[1]	\f$\eta\f$<br>
 */
template<unsigned int spacedim>
class OmegaMixedTerm02 : public incrementalFE::Omega<spacedim-1, spacedim>
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
	OmegaMixedTerm02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						const unsigned int																method,
						const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 1, 0, 1, 0, method, alpha, "OmegaMixedTerm02")
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

		const double I_dot = values[0];
		const double eta = values[1];

		if(get<0>(requested_quantities))
		{
			sigma = -I_dot * eta;
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = -eta;
			d_sigma[1] = -I_dot;
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma(0,1) = d2_sigma(1,0) = -1.0;
		}

		return false;
	}

};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	-\mu ( \nabla \cdot \dot{\boldsymbol{I}} + \dot{c} ) \f$
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
 * \f$ \omega^\Sigma =	-\mu \dot{\boldsymbol{I}} \cdot \boldsymbol{n} \f$
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
 * \f$ \omega^\Sigma =	\bar\mu(t) \dot{\boldsymbol{I}} \cdot \boldsymbol{n} \f$
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
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 3, 0, 0, 0, method, alpha, "OmegaFluxPower00"),
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
 * \f$ \omega^\Sigma =	-\bar i(t) \eta \f$
 *
 * where \f$\bar i\f$ is a prescrobed normal flux, and \f$\eta\f$ the corresponding potential.
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\eta\f$<br>
 */
template<unsigned int spacedim>
class OmegaDualFluxPower00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * %Function determining \f$\bar i(t)\f$
	 */
	dealii::Function<spacedim>&
	function_i;

public:

	/**
	 * whether this scalar functional is active
	 */
	bool
	is_active = true;


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
	 * @param[in]		function_i				OmegaDualFluxPower00::function_i
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaDualFluxPower00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							dealii::Function<spacedim>&														function_i,
							const unsigned int																method,
							const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 0, 0, 1, 0, method, alpha, "OmegaDualFluxPower00"),
	function_i(function_i)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			/*d2_sigma*/,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{
		if(is_active)
		{
			const double time_old = function_i.get_time();
			function_i.set_time(t);
			const double i_bar = function_i.value(x);
			function_i.set_time(time_old);

			const double eta = values[0];

			if(get<0>(requested_quantities))
				sigma = -i_bar * eta;

			if(get<1>(requested_quantities))
				d_sigma[0] = - i_bar;
		}

		return false;
	}

};


/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	-\bar{\boldsymbol{f}}(t) \dot{\boldsymbol{u}} \f$
 *
 * where \f$\bar{\boldsymbol{f}}\f$ is the prescribed traction, and \f$\dot{\boldsymbol{u}}\f$ the corresponding displacement.
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\dot{u}_x\f$<br>
 * 															[1]	\f$\dot{u}_y\f$<br>
 * 															[2]	\f$\dot{u}_z\f$<br>
 */
template<unsigned int spacedim>
class OmegaTraction00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * %Function determining \f$\bar{\boldsymbol{f}}(t)\f$ (must have three components)
	 */
	dealii::Function<spacedim>&
	function_f;

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
	 * @param[in]		function_f				OmegaTraction00::function_f
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 */
	OmegaTraction00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						dealii::Function<spacedim>&														function_f,
						const unsigned int																method,
						const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 0, 3, 0, 0, method, alpha, "OmegaTraction00"),
	function_f(function_f)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			/*d2_sigma*/,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{

		const double time_old = function_f.get_time();
		function_f.set_time(t);
		const double f_x = function_f.value(x, 0);
		const double f_y = function_f.value(x, 1);
		const double f_z = function_f.value(x, 2);
		function_f.set_time(time_old);

		const double u_x_dot = values[0];
		const double u_y_dot = values[1];
		const double u_z_dot = values[2];

		if(get<0>(requested_quantities))
		{
			sigma = - (f_x * u_x_dot + f_y * u_y_dot + f_z * u_z_dot);
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = -f_x;
			d_sigma[1] = -f_y;
			d_sigma[2] = -f_z;
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	1/(2D) \left(\dot{\boldsymbol{I}} \cdot \boldsymbol{n}\right)^2 \f$
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
 * \f$ \omega^\Sigma =	-I_0\left[ \dfrac{1}{1-\beta} \exp\left( - \dfrac{1-\beta}{RT} \Delta \eta \right) + \dfrac{1}{\beta} \exp\left( \dfrac{\beta}{RT} \Delta \eta \right) \right] \f$
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
 * \f$ \omega^\Sigma =	\boldsymbol{t} \cdot \left( \boldsymbol{u}^0 - \boldsymbol{u}^1 \right) \f$
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
 * \f$ \omega^\Omega =	-\dfrac{D c J}{2 c^\mathrm{f} V^\mathrm{f}_\mathrm{m}} \nabla\eta \cdot \boldsymbol{C}^{-1} \cdot \nabla\eta - \eta \dot{c} \f$,
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
		const double J = determinant(F);
		const double n_f = c_f / J * V_m_f;
		const dealii::Tensor<1, 3> A = C_inv * grad_eta;
		const double grad_eta_C_inv_grad_eta = A * grad_eta;

		if(get<0>(requested_quantities))
		{
			omega = -0.5 * D * c / n_f * grad_eta_C_inv_grad_eta - eta * c_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = -eta;
			for(unsigned int m = 0; m < 3; ++m)
				d_omega[m + 1] = -D * c / n_f * A[m];
			d_omega[4] = -c_dot;
		}

		if(get<2>(requested_quantities))
		{
			d2_omega = 0.0;
			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int n = 0; n < 3; ++n)
					d2_omega(m + 1, n + 1) = -D * c / n_f * C_inv[m][n];
			d2_omega(0, 4) = d2_omega(4, 0) = -1.0;

			if(compute_dq)
			{
				const dealii::Tensor<1,3> F_A = F * A;
				const dealii::Tensor<2,3> F_C_inv = F * C_inv;
				dealii::Vector<double> F_vect(9), dJ_dF(9);
				for(unsigned i = 0; i < 9; ++i)
					F_vect[i] = values[7 + i];
				get_dJ_dF(F_vect, dJ_dF);

				for(unsigned int L = 0; L < 3; ++L)
				{
					for(unsigned int m = 0; m < 3; ++m)
						for(unsigned int M = 0; M < 3; ++M)
							d2_omega[1 + L][7 + m * 3 + M] = (- D / V_m_f * A[L] * dJ_dF[m * 3 + M] + D * J / V_m_f * (C_inv[L][M] * F_A[m] + A[M] * F_C_inv[m][L] ) ) * c / c_f;
					d2_omega[1 + L][5] = - D * J / V_m_f * 1.0 / c_f * A[L];
					d2_omega[1 + L][6] = D * J / V_m_f * c / c_f / c_f * A[L];
				}
			}


		}

		return false;

	}

};


/**
 * Class defining dual dissipation associated with flux of ions through fluid, which itself possibly flows through solid skeleton.
 *
 * \f$ \omega^\Omega =	-\dfrac{D c J}{2 c^\mathrm{f} V^\mathrm{f}_\mathrm{m}} \nabla\eta \cdot \boldsymbol{C}^{-1} \cdot \nabla\eta - \eta \dot{c} \f$,
 *
 * where \f$c\f$ is the species concentration,<br>
 * \f$c^\mathrm{f} = \dfrac{J-n_0}{V^\mathrm{f}_\mathrm{m}}\f$ the fluid concentration, with \f$n_0\f$ being a material parameter,<br>
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
 * 																				[6]  \f$F_{xx}\f$<br>
 * 																				[7]  \f$F_{xy}\f$<br>
 * 																				[8]  \f$F_{xz}\f$<br>
 * 																				[9]  \f$F_{yx}\f$<br>
 * 																				[10] \f$F_{yy}\f$<br>
 * 																				[11] \f$F_{yz}\f$<br>
 * 																				[12] \f$F_{zx}\f$<br>
 * 																				[13] \f$F_{zy}\f$<br>
 * 																				[14] \f$F_{zz}\f$<br>
 */
template<unsigned int spacedim>
class OmegaDualIonDissipation01 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * mobility \f$D\f$
	 */
	const double
	D;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

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
	 * @param[in]		D						OmegaDualIonDissipation01::D
	 *
	 * @param[in]		n_0						OmegaDualIonDissipation01::n_0
	 *
	 * @param[in]		V_m_f					OmegaDualIonDissipation01::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualIonDissipation01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																D,
								const double																n_0,
								const double																V_m_f,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 0, 4, 10, method, alpha, "OmegaDualIonDissipation01"),
	D(D),
	n_0(n_0),
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
		dealii::Tensor<2, 3> F, C, C_inv;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[6 + m * 3 + n];
		C = transpose(F) * F;
		C_inv = invert(C);
		const double J = determinant(F);
		const double c_f = (J - n_0) / V_m_f;
		const double n_f = c_f / J * V_m_f;
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
			d2_omega = 0.0;
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
 * \f$ \omega^\Omega =	-\dfrac{D J}{2 V^\mathrm{f}_\mathrm{m}}  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) \cdot \boldsymbol{C}^{-1} \cdot \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
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

		dealii::Tensor<1,3> grad_eta, grad_eta_f;
		std::vector<dealii::Tensor<1,3>> grad_eta_i(I);
		for(unsigned int m = 0; m < 3; ++m)
		{
			grad_eta[m] = values[i_grad_eta_f + m];
			grad_eta_f[m] = values[i_grad_eta_f + m];
		}
		for(unsigned int i = 0; i < I; ++i)
		{
			for(unsigned int m = 0; m < 3; ++m)
			{
				grad_eta[m] += c_i[i]/c_f * values[i_grad_eta_i[i] + m];
				grad_eta_i[i][m] = values[i_grad_eta_i[i] + m];
			}
		}

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2, 3> F, C, C_inv;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		C = transpose(F) * F;
		C_inv = invert(C);
		const dealii::Tensor<1, 3> C_inv_grad_eta = C_inv * grad_eta;
		const double grad_eta_C_inv_grad_eta = C_inv_grad_eta * grad_eta;
		const double J = determinant(F);

		const double  K = D * J / V_m_f;

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
			d2_omega = 0.0;

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

			if(compute_dq)
			{
				const dealii::Tensor<1,3> A = C_inv * grad_eta;
				const dealii::Tensor<1,3> F_A = F * A;
				const dealii::Tensor<2,3> F_C_inv = F * C_inv;
				dealii::Vector<double> F_vect(9), dJ_dF(9);
				for(unsigned i = 0; i < 9; ++i)
					F_vect[i] = values[i_F + i];
				get_dJ_dF(F_vect, dJ_dF);
				dealii::Tensor<1,3> C_inv_sum_c_i_c_f_grad_eta_i;
				for(unsigned int L = 0; L < 3; ++L)
					for(unsigned int i = 0; i < I; ++i)
						C_inv_sum_c_i_c_f_grad_eta_i[L] += c_i[i]/c_f * grad_eta_i[i][L];
				C_inv_sum_c_i_c_f_grad_eta_i = C_inv * C_inv_sum_c_i_c_f_grad_eta_i;

				for(unsigned int L = 0; L < 3; ++L)
				{
					for(unsigned int m = 0; m < 3; ++m)
					{
						for(unsigned int M = 0; M < 3; ++M)
						{
							d2_omega[i_grad_eta_f + L][i_F + m * 3 + M] = - D / V_m_f * A[L] * dJ_dF[m * 3 + M] + K * (C_inv[L][M] * F_A[m] + A[M] * F_C_inv[m][L] );
							for(unsigned int i = 0; i < I; ++i)
								d2_omega[i_grad_eta_i[i] + L][i_F + m * 3 + M] = d2_omega[i_grad_eta_f + L][i_F + m * 3 + M] * c_i[i] / c_f;
						}
					}
					d2_omega[i_grad_eta_f + L][i_c_f] = 1.0 / c_f * K * C_inv_sum_c_i_c_f_grad_eta_i[L];
					for(unsigned int i = 0; i < I; ++i)
						d2_omega[i_grad_eta_i[i] + L][i_c_f] = K * (C_inv_sum_c_i_c_f_grad_eta_i[L] + A[L]) * c_i[i] / c_f / c_f;
					for(unsigned int j = 0; j < I; ++j)
					{
						const dealii::Tensor<1,3> C_inv_grad_eta_j = C_inv * grad_eta_i[j];
						d2_omega[i_grad_eta_f + L][i_c_i[j]] = -K * C_inv_grad_eta_j[L] / c_f;
						for(unsigned int i = 0; i < I; ++i)
							d2_omega[i_grad_eta_i[i] + L][i_c_i[j]] = -K * C_inv_grad_eta_j[L] * c_i[i] / c_f / c_f;
						d2_omega[i_grad_eta_i[j] + L][i_c_i[j]] += -K * A[L] / c_f;
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
 * \f$ \omega^\Omega =	\nabla \dot{\xi} \cdot  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
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
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 3, 4+3*I, 1+I, method, alpha, "OmegaDualFluidDissipation01"),
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


		const double eps = 0.0;
		if(get<0>(requested_quantities))
		{
			omega = grad_xi_dot * grad_eta - eta_f * c_f_dot + grad_xi_dot * grad_xi_dot * 0.5 * eps;
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
				d_omega[i_grad_xi_dot + m] += grad_xi_dot[m] * eps;
			}
		}

		if(get<2>(requested_quantities))
		{
			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int m = 0; m < 3; ++m)
			{
				d2_omega[i_grad_xi_dot + m][i_grad_eta_f + m] = d2_omega[i_grad_eta_f + m][i_grad_xi_dot + m] = 1.0;
				for(unsigned int i = 0; i < I; ++i)
					d2_omega[i_grad_xi_dot + m][i_grad_eta_i[i] + m] = d2_omega[i_grad_eta_i[i] + m][i_grad_xi_dot + m] = c_i[i]/c_f;
			}

			if(compute_dq)
			{
				for(unsigned int m = 0; m < 3; ++m)
				{
					d2_omega[i_grad_xi_dot + m][i_c_f] = (values[i_grad_eta_f + m] - grad_eta[m]) / c_f;
					for(unsigned int i = 0; i < I; ++i)
						d2_omega[i_grad_xi_dot + m][i_c_i[i]] = values[i_grad_eta_i[i] + m] / c_f;
					for(unsigned int i = 0; i < I; ++i)
					{
						d2_omega[i_grad_eta_i[i] + m][i_c_f] = -grad_xi_dot[m] * c_i[i]/c_f/c_f;
						d2_omega[i_grad_eta_i[i] + m][i_c_i[i]] = grad_xi_dot[m] * 1.0/c_f;
					}
				}

			}

			for(unsigned int m = 0; m < 3; ++m)
			{
				d2_omega(i_grad_xi_dot + m, i_grad_xi_dot + m) += eps;
			}

		}

		return false;

	}
};


/**
 * Class defining dual dissipation associated with flux of fluid through a solid skeleton, where ions flow in addition through the fluid.
 *
 * \f$ \omega^\Omega =	-\dfrac{D J}{2 V^\mathrm{f}_\mathrm{m}}  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) \cdot \boldsymbol{C}^{-1} \cdot \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$c^\mathrm{f} = \dfrac{J - n_0}{V^\mathrm{f}_\mathrm{m}}\f$ is the fluid concentration, with \f$ n_0\f$ being a material parameter.<br>
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
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  				\f$\dot{F}_{xx}\f$<br>
 * 																				[1]  				\f$\dot{F}_{xy}\f$<br>
 * 																				[2]  				\f$\dot{F}_{xz}\f$<br>
 * 																				[3]  				\f$\dot{F}_{yx}\f$<br>
 * 																				[4]  				\f$\dot{F}_{yy}\f$<br>
 * 																				[5]  				\f$\dot{F}_{yz}\f$<br>
 * 																				[6]  				\f$\dot{F}_{zx}\f$<br>
 * 																				[7]  				\f$\dot{F}_{zy}\f$<br>
 * 																				[8]  				\f$\dot{F}_{zz}\f$<br>
 * 																				[9]  				\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[10]  				\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[11]  				\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[12] ... [11+3I]	\f$\eta^i_{,x}\f$, \f$\eta^i_{,y}\f$, \f$\eta^i_{,z}\f$ (ordering: xyz, xyz, ...)<br>
 * 																				[12+3I]				\f$\eta^\mathrm{f}\f$<br>
 * 																				[13+3I] ... [12+4I]	\f$c^i\f$<br>
 * 																				[13+4I] ... [21+4I]	\f$F_{xx}\f$, \f$F_{xy}\f$, \f$F_{xz}\f$, \f$F_{yx}\f$, \f$F_{yy}\f$, \f$F_{yz}\f$, \f$F_{zx}\f$, \f$F_{zy}\f$, \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaDualFluidDissipation02 : public incrementalFE::Omega<spacedim, spacedim>
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
	 * \f$n_0\f$
	 */
	const double
	n_0;

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
	 * @param[in]		I						OmegaDualFluidDissipation02::I
	 *
	 * @param[in]		D						OmegaDualFluidDissipation02::D
	 *
	 * @param[in]		n_0						OmegaDualFluidDissipation02::n_0
	 *
	 * @param[in]		V_m_f					OmegaDualFluidDissipation02::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualFluidDissipation02(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															I,
								const double																D,
								const double																n_0,
								const double																V_m_f,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 9, 0, 4+3*I, 9+I, method, alpha, "OmegaDualFluidDissipation02"),
	I(I),
	D(D),
	n_0(n_0),
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
		const unsigned int i_F_dot = 0;
		const unsigned int i_grad_eta_f = 9;
		vector<unsigned int> i_grad_eta_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_grad_eta_i[i] = 12 + 3*i;
		const unsigned int i_eta_f = 12 + 3*I;
		vector<unsigned int> i_c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_c_i[i] = 13 + 3*I + i;
		const unsigned int i_F = 13 + 4*I;

		dealii::Tensor<2, 3> F_dot, F;
		for(unsigned int m = 0; m < 3; ++m)
		{
			for(unsigned int n = 0; n < 3; ++n)
			{
				F_dot[m][n] = values [i_F_dot + m * 3 + n];
				F[m][n] = values [i_F + m * 3 + n];
			}
		}
		dealii::Vector<double> c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			c_i[i] = values[i_c_i[i]];
		const double J =determinant(F);
		const double c_f = (J - n_0) / V_m_f;

		dealii::Tensor<1,3> grad_eta;
		for(unsigned int m = 0; m < 3; ++m)
			grad_eta[m] = values[i_grad_eta_f + m];
		for(unsigned int i = 0; i < I; ++i)
		{
			for(unsigned int m = 0; m < 3; ++m)
			{
				grad_eta[m] += c_i[i]/c_f * values[i_grad_eta_i[i] + m];
			}
		}

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2, 3> F_inv, C_inv;
		F_inv = invert(F);
		C_inv = F_inv * transpose(F_inv);
		const dealii::Tensor<1, 3> C_inv_grad_eta = C_inv * grad_eta;
		const double grad_eta_C_inv_grad_eta = C_inv_grad_eta * grad_eta;

		const double  K = D * J / V_m_f;

		if(get<0>(requested_quantities))
		{
			omega = -0.5 * K * grad_eta_C_inv_grad_eta - eta_f * J * scalar_product(F_inv, transpose(F_dot)) / V_m_f;
		}

		if(get<1>(requested_quantities))
		{
			d_omega = 0.0;
			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int n = 0; n < 3; ++n)
					d_omega[i_F_dot + m * 3 + n] = -eta_f * J / V_m_f * F_inv[n][m];
			d_omega[i_eta_f]   = -J * scalar_product(F_inv, transpose(F_dot)) / V_m_f;
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_omega[i_grad_eta_f + m] = -K * C_inv_grad_eta[m];
				for(unsigned int i = 0; i < I; ++i)
					d_omega[i_grad_eta_i[i] + m] = d_omega[i_grad_eta_f + m] * c_i[i]/c_f;
			}
		}

		if(get<2>(requested_quantities))
		{
			d2_omega = 0.0;
			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int n = 0; n < 3; ++n)
					d2_omega[i_F_dot + m * 3 + n][i_eta_f] = d2_omega[i_eta_f][i_F_dot + m * 3 + n] = -J / V_m_f * F_inv[n][m];

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
 * \f$ \omega^\Omega =	\nabla \dot{\xi} \cdot  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$\xi\f$ is a scalar potential,<br>
 * \f$c^\mathrm{f} = \dfrac{J-n_0}{V^\mathrm{f}_\mathrm{m}}\f$ the fluid concentration with \f$J=\det\boldsymbol{F}\f$ being the determinant of the deformation gradient  \f$\boldsymbol{F}\f$ and \f$n_0\f$ and \f$V^\mathrm{f}_\mathrm{m}\f$ being parameters,<br>
 * \f$c^i\f$ are the ion concentrations (\f$i\f$ runs from \f$1\f$ to \f$I\f$),<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential,<br>
 * and \f$\eta^i\f$ are the ion potentials
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  				\f$\dot{\xi}_{,x}\f$<br>
 * 																				[1]					\f$\dot{\xi}_{,y}\f$<br>
 * 																				[2]					\f$\dot{\xi}_{,z}\f$<br>
 *																				[3]  				\f$\dot{F}_{xx}\f$<br>
 * 																				[4]  				\f$\dot{F}_{xy}\f$<br>
 * 																				[5]  				\f$\dot{F}_{xz}\f$<br>
 * 																				[6]  				\f$\dot{F}_{yx}\f$<br>
 * 																				[7]  				\f$\dot{F}_{yy}\f$<br>
 * 																				[8]  				\f$\dot{F}_{yz}\f$<br>
 * 																				[9]  				\f$\dot{F}_{zx}\f$<br>
 * 																				[10]  				\f$\dot{F}_{zy}\f$<br>
 * 																				[11]  				\f$\dot{F}_{zz}\f$<br>
 * 																				[12]  				\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[13]  				\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[14]  				\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[15] ... [14+3I]	\f$\eta^i_{,x}\f$, \f$\eta^i_{,y}\f$, \f$\eta^i_{,z}\f$ (ordering: xyz, xyz, ...)<br>
 * 																				[15+3I]				\f$\eta^\mathrm{f}\f$<br>
 * 																				[16+3I] ... [24+3I]	\f$F_{xx}\f$, \f$F_{xy}\f$, \f$F_{xz}\f$, \f$F_{yx}\f$, \f$F_{yy}\f$, \f$F_{yz}\f$, \f$F_{zx}\f$, \f$F_{zy}\f$, \f$F_{zz}\f$<br>
 * 																				[25+3I] ... [24+4I]	\f$c^i\f$<br>
 */
template<unsigned int spacedim>
class OmegaDualFluidDissipation03 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * Number of ionic species \f$I\f$
	 */
	const unsigned int
	I;

	/**
	 * \f$n_0\f$
	 */
	const double
	n_0;

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
	 * @param[in]		I						OmegaDualFluidDissipation03::I
	 *
	 * @param[in]		n_0						OmegaDualFluidDissipation03::n_0
	 *
	 * @param[in]		V_m_f					OmegaDualFluidDissipation03::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaDualFluidDissipation03(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															I,
								const double																n_0,
								const double																V_m_f,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 9, 3, 4+3*I, 9+I, method, alpha, "OmegaDualFluidDissipation03"),
	I(I),
	n_0(n_0),
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
		const unsigned int i_grad_xi_dot = 0;
		const unsigned int i_F_dot = 3;
		const unsigned int i_grad_eta_f = 12;
		vector<unsigned int> i_grad_eta_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_grad_eta_i[i] = 15 + 3*i;
		const unsigned int i_eta_f = 15 + 3*I;
		const unsigned int i_F = 16 + 3*I;
		vector<unsigned int> i_c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_c_i[i] = 25 + 3*I + i;

		dealii::Tensor<1,3> grad_xi_dot;
		for(unsigned int m = 0; m < 3; ++m)
			grad_xi_dot[m] = values[i_grad_xi_dot + m];

		dealii::Tensor<2, 3> F_dot, F;
		for(unsigned int m = 0; m < 3; ++m)
		{
			for(unsigned int n = 0; n < 3; ++n)
			{
				F_dot[m][n] = values [i_F_dot + m * 3 + n];
				F[m][n] = values [i_F + m * 3 + n];
			}
		}

		dealii::Vector<double> c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			c_i[i] = values[i_c_i[i]];

		const double J =determinant(F);
		const double c_f = (J - n_0) / V_m_f;

		dealii::Tensor<1,3> grad_eta;
		for(unsigned int m = 0; m < 3; ++m)
			grad_eta[m] = values[i_grad_eta_f + m];
		for(unsigned int i = 0; i < I; ++i)
			for(unsigned int m = 0; m < 3; ++m)
				grad_eta[m] += c_i[i]/c_f * values[i_grad_eta_i[i] + m];

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2, 3> F_inv;
		F_inv = invert(F);

		if(get<0>(requested_quantities))
		{
			omega = grad_xi_dot * grad_eta - eta_f * J * scalar_product(F_inv, transpose(F_dot)) / V_m_f;
		}

		if(get<1>(requested_quantities))
		{
			d_omega = 0.0;

			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int n = 0; n < 3; ++n)
					d_omega[i_F_dot + m * 3 + n] = -eta_f * J / V_m_f * F_inv[n][m];
			d_omega[i_eta_f]   = -J * scalar_product(F_inv, transpose(F_dot)) / V_m_f;
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
			d2_omega = 0.0;
			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int n = 0; n < 3; ++n)
					d2_omega[i_F_dot + m * 3 + n][i_eta_f] = d2_omega[i_eta_f][i_F_dot + m * 3 + n] = -J / V_m_f * F_inv[n][m];

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

/**
 * Class defining Lagrangian multiplier term for equilibrium condition in the case that fluid flows without dissipation, where ions flow in addition with dissipation through the fluid.
 *
 * \f$ \omega^\Omega =	\left[ c^\mathrm{f} \boldsymbol{F}^{-1} \cdot \left( \nabla \dot{\xi} \cdot \boldsymbol{F}^{-1} - \dot{\boldsymbol{u}} \right) \right] \cdot  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$\xi\f$ is a scalar potential,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * \f$c^i\f$ are the ion concentrations (\f$i\f$ runs from \f$1\f$ to \f$I\f$),<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential,<br>
 * and \f$\eta^i\f$ are the ion potentials
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  					\f$\dot{\xi}_{,x}\f$<br>
 * 																				[1]						\f$\dot{\xi}_{,y}\f$<br>
 * 																				[2]						\f$\dot{\xi}_{,z}\f$<br>
 *																				[3]  					\f$\dot{c}^\mathrm{f}\f$<br>
 *																				[4]						\f$\dot{u}_x\f$<br>
 *																				[5]						\f$\dot{u}_y\f$<br>
 *																				[6]						\f$\dot{u}_z\f$<br>
 * 																				[7]  					\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[8]  					\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[9]  					\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[10] ... [9+3I]			\f$\eta^i_{,x}\f$, \f$\eta^i_{,y}\f$, \f$\eta^i_{,z}\f$ (ordering: xyz, xyz, ...)<br>
 * 																				[10+3I]					\f$\eta^\mathrm{f}\f$<br>
 * 																				[11+3I]					\f$c^\mathrm{f}\f$<br>
 * 																				[12+3I] ... [11+4I]		\f$c^i\f$<br>
 * 																				[11+4I+1] ... [11+4I+9]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaDualFluidDissipation04 : public incrementalFE::Omega<spacedim, spacedim>
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
	OmegaDualFluidDissipation04(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															I,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 4, 3, 4+3*I, 10+I, method, alpha, "OmegaDualFluidDissipation04"),
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

		// start indices for respective quantities
		const unsigned int i_grad_xi_dot = 0;
		const unsigned int i_c_f_dot = 3;
		const unsigned int i_u_dot = 4;
		const unsigned int i_grad_eta_f = 7;
		vector<unsigned int> i_grad_eta_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_grad_eta_i[i] = 10 + 3*i;
		const unsigned int i_eta_f = 10 + 3*I;
		const unsigned int i_c_f = 11 + 3*I;
		vector<unsigned int> i_c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_c_i[i] = 12 + 3*I + i;
		const unsigned int i_F = 11+ 4*I + 1;

		dealii::Tensor<1,3> grad_xi_dot;
		for(unsigned int m = 0; m < 3; ++m)
			grad_xi_dot[m] = values[i_grad_xi_dot + m];

		const double c_f_dot = values[i_c_f_dot];

		dealii::Tensor<1,3> u_dot;
		for(unsigned int m = 0; m < 3; ++m)
			u_dot[m] = values[i_u_dot + m];

		const double eta_f = values[i_eta_f];

		const double c_f = values[i_c_f];
		dealii::Vector<double> c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			c_i[i] = values[i_c_i[i]];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];

		dealii::Tensor<1,3> grad_eta_c_f, grad_eta_f;
		std::vector<dealii::Tensor<1,3>> grad_eta_i(I);
		for(unsigned int m = 0; m < 3; ++m)
		{
			grad_eta_c_f[m] = c_f * values[i_grad_eta_f + m];
			grad_eta_f[m] = values[i_grad_eta_f + m];
		}
		for(unsigned int i = 0; i < I; ++i)
		{
			for(unsigned int m = 0; m < 3; ++m)
			{
				grad_eta_c_f[m] += c_i[i] * values[i_grad_eta_i[i] + m];
				grad_eta_i[i][m] = values[i_grad_eta_i[i] + m];
			}
		}

		dealii::Tensor<2,3> F_inv, C_inv;
		dealii::Tensor<1,3> C_inv_grad_xi_dot, grad_xi_dot_F_inv, I_dot_f_c_f, C_inv_grad_eta_c_f, grad_eta_c_f_F_inv, C_inv_grad_eta_f, F_inv_grad_eta_f, F_inv_u_dot;
		std::vector<dealii::Tensor<1,3>> C_inv_grad_eta_i(I), F_inv_grad_eta_i(I);
		F_inv = invert(F);
		C_inv = F_inv * transpose(F_inv);
		C_inv_grad_xi_dot = C_inv * grad_xi_dot;
		grad_xi_dot_F_inv = transpose(F_inv) * grad_xi_dot;
		I_dot_f_c_f = F_inv * (grad_xi_dot_F_inv - u_dot);
		C_inv_grad_eta_c_f = C_inv * grad_eta_c_f;
		grad_eta_c_f_F_inv = transpose(F_inv) * grad_eta_c_f;
		C_inv_grad_eta_f = C_inv * grad_eta_f;
		F_inv_grad_eta_f = transpose(F_inv) * grad_eta_f;
		F_inv_u_dot = F_inv * u_dot;
		for(unsigned int i = 0; i < I; ++i)
		{
			C_inv_grad_eta_i[i] = C_inv * grad_eta_i[i];
			F_inv_grad_eta_i[i] = transpose(F_inv) * grad_eta_i[i];
		}

		if(get<0>(requested_quantities))
		{
			omega = I_dot_f_c_f * grad_eta_c_f - eta_f * c_f_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_c_f_dot] = -eta_f;
			d_omega[i_eta_f]   = -c_f_dot;
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_omega[i_grad_xi_dot + m] = C_inv_grad_eta_c_f[m];
				d_omega[i_u_dot + m] = -grad_eta_c_f_F_inv[m];
				d_omega[i_grad_eta_f + m] = I_dot_f_c_f [m] * c_f;
				for(unsigned int i = 0; i < I; ++i)
					d_omega[i_grad_eta_i[i] + m] = I_dot_f_c_f[m] * c_i[i];
			}
		}

		if(get<2>(requested_quantities))
		{

			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int m = 0; m < 3; ++m)
			{
				for(unsigned int n = 0; n < 3; ++n)
				{
					d2_omega(i_grad_xi_dot + m, i_grad_eta_f + n) = d2_omega(i_grad_eta_f + n, i_grad_xi_dot + m) = C_inv[m][n] * c_f;
					d2_omega(i_u_dot + m, i_grad_eta_f + n) = d2_omega(i_grad_eta_f + n, i_u_dot + m) = -F_inv[n][m] * c_f;
				}
				for(unsigned int i = 0; i < I; ++i)
				{
					for(unsigned int n = 0; n < 3; ++n)
					{
						d2_omega(i_grad_xi_dot + m, i_grad_eta_i[i] + n) = d2_omega(i_grad_eta_i[i] + n, i_grad_xi_dot + m) = C_inv[m][n] * c_i[i];
						d2_omega(i_u_dot + m, i_grad_eta_i[i] + n) = d2_omega(i_grad_eta_i[i] + n, i_u_dot + m) = -F_inv[n][m] * c_i[i];
					}
				}

			}


			if(compute_dq)
			{
				for(unsigned int m = 0; m < 3; ++m)
				{
					d2_omega(i_grad_xi_dot + m, i_c_f) = C_inv_grad_eta_f[m];
					d2_omega(i_u_dot + m, i_c_f) = -F_inv_grad_eta_f[m];
					d2_omega(i_grad_eta_f + m, i_c_f) = d_omega(i_grad_eta_f + m) / c_f;
					for(unsigned int i = 0; i < I; ++i)
					{
						d2_omega(i_grad_xi_dot + m, i_c_i[i]) = C_inv_grad_eta_i[i][m];
						d2_omega(i_u_dot + m, i_c_i[i]) = -F_inv_grad_eta_i[i][m];
						d2_omega(i_grad_eta_i[i] + m, i_c_i[i]) = d_omega(i_grad_eta_i[i] + m) / c_i[i];
 					}
					for(unsigned int k = 0; k < 3; ++k)
					{
						for(unsigned int l = 0; l < 3; ++l)
						{
							d2_omega(i_grad_xi_dot + m, i_F + 3 * k + l) = -C_inv_grad_eta_c_f[l] * F_inv[m][k] - grad_eta_c_f_F_inv[k] * C_inv[l][m];
							d2_omega(i_u_dot + m, i_F + 3 * k + l) = grad_eta_c_f_F_inv[k] * F_inv[l][m];
							d2_omega(i_grad_eta_f + m, i_F + 3 * k + l) = (-C_inv_grad_xi_dot[l] * F_inv[m][k] - grad_xi_dot_F_inv[k] * C_inv[l][m] + F_inv_u_dot[l] * F_inv[m][k]) * c_f;
							for(unsigned int i = 0; i < I; ++i)
								d2_omega(i_grad_eta_i[i] + m, i_F + 3 * k + l) = (-C_inv_grad_xi_dot[l] * F_inv[m][k] - grad_xi_dot_F_inv[k] * C_inv[l][m] + F_inv_u_dot[l] * F_inv[m][k]) * c_i[i];

						}
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
 * \f$ \omega^\Omega =	\left[ c^\mathrm{f} \boldsymbol{F}^{-1} \cdot \left( \dot{\boldsymbol{v}} - \dot{\boldsymbol{u}} \right) \right] \cdot  \left(\nabla \eta^\mathrm{f} + \sum^I_{i=1} \dfrac{c^i}{c^\mathrm{f}} \nabla\eta^i \right) - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * \f$c^i\f$ are the ion concentrations (\f$i\f$ runs from \f$1\f$ to \f$I\f$),<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential,<br>
 * and \f$\eta^i\f$ are the ion potentials
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  					\f$\dot{v}_{x}\f$<br>
 * 																				[1]						\f$\dot{v}_{y}\f$<br>
 * 																				[2]						\f$\dot{v}_{z}\f$<br>
 *																				[3]  					\f$\dot{c}^\mathrm{f}\f$<br>
 *																				[4]						\f$\dot{u}_x\f$<br>
 *																				[5]						\f$\dot{u}_y\f$<br>
 *																				[6]						\f$\dot{u}_z\f$<br>
 * 																				[7]  					\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[8]  					\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[9]  					\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[10] ... [9+3I]			\f$\eta^i_{,x}\f$, \f$\eta^i_{,y}\f$, \f$\eta^i_{,z}\f$ (ordering: xyz, xyz, ...)<br>
 * 																				[10+3I]					\f$\eta^\mathrm{f}\f$<br>
 * 																				[11+3I]					\f$c^\mathrm{f}\f$<br>
 * 																				[12+3I] ... [11+4I]		\f$c^i\f$<br>
 * 																				[11+4I+1] ... [11+4I+9]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaDualFluidDissipation05 : public incrementalFE::Omega<spacedim, spacedim>
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
	OmegaDualFluidDissipation05(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															I,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 4, 3, 4+3*I, 10+I, method, alpha, "OmegaDualFluidDissipation05"),
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

		// start indices for respective quantities
		const unsigned int i_v_dot = 0;
		const unsigned int i_c_f_dot = 3;
		const unsigned int i_u_dot = 4;
		const unsigned int i_grad_eta_f = 7;
		std::vector<unsigned int> i_grad_eta_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_grad_eta_i[i] = 10 + 3*i;
		const unsigned int i_eta_f = 10 + 3*I;
		const unsigned int i_c_f = 11 + 3*I;
		std::vector<unsigned int> i_c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			i_c_i[i] = 12 + 3*I + i;
		const unsigned int i_F = 11+ 4*I + 1;

		dealii::Tensor<1,3> v_dot;
		for(unsigned int m = 0; m < 3; ++m)
			v_dot[m] = values[i_v_dot + m];

		const double c_f_dot = values[i_c_f_dot];

		dealii::Tensor<1,3> u_dot;
		for(unsigned int m = 0; m < 3; ++m)
			u_dot[m] = values[i_u_dot + m];

		const double eta_f = values[i_eta_f];

		const double c_f = values[i_c_f];
		dealii::Vector<double> c_i(I);
		for(unsigned int i = 0; i < I; ++i)
			c_i[i] = values[i_c_i[i]];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];

		dealii::Tensor<1,3> grad_eta_c_f, grad_eta_f;
		std::vector<dealii::Tensor<1,3>> grad_eta_i(I);
		for(unsigned int m = 0; m < 3; ++m)
		{
			grad_eta_c_f[m] = c_f * values[i_grad_eta_f + m];
			grad_eta_f[m] = values[i_grad_eta_f + m];
		}
		for(unsigned int i = 0; i < I; ++i)
		{
			for(unsigned int m = 0; m < 3; ++m)
			{
				grad_eta_c_f[m] += c_i[i] * values[i_grad_eta_i[i] + m];
				grad_eta_i[i][m] = values[i_grad_eta_i[i] + m];
			}
		}

		dealii::Tensor<2,3> F_inv;
		dealii::Tensor<1,3> F_inv_delta_v_dot, grad_eta_f_F_inv, grad_eta_c_f_F_inv, delta_v_dot;
		std::vector<dealii::Tensor<1,3>> grad_eta_i_F_inv(I);
		F_inv = invert(F);
		delta_v_dot = v_dot - u_dot;
		F_inv_delta_v_dot = F_inv * delta_v_dot;
		grad_eta_f_F_inv = transpose(F_inv) * grad_eta_f;
		grad_eta_c_f_F_inv = transpose(F_inv) * grad_eta_c_f;
		for(unsigned int i = 0; i < I; ++i)
			grad_eta_i_F_inv[i] = transpose(F_inv) * grad_eta_i[i];

		if(get<0>(requested_quantities))
		{
			omega = F_inv_delta_v_dot * grad_eta_c_f - eta_f * c_f_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_c_f_dot] = -eta_f;
			d_omega[i_eta_f]   = -c_f_dot;
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_omega[i_v_dot + m] = grad_eta_c_f_F_inv[m];
				d_omega[i_u_dot + m] = -grad_eta_c_f_F_inv[m];
				d_omega[i_grad_eta_f + m] = F_inv_delta_v_dot[m] * c_f;
				for(unsigned int i = 0; i < I; ++i)
					d_omega[i_grad_eta_i[i] + m] = F_inv_delta_v_dot[m] * c_i[i];
			}
		}

		if(get<2>(requested_quantities))
		{

			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int m = 0; m < 3; ++m)
			{
				for(unsigned int n = 0; n < 3; ++n)
				{
					d2_omega(i_v_dot + m, i_grad_eta_f + n) = d2_omega(i_grad_eta_f + n, i_v_dot + m) = F_inv[n][m] * c_f;
					d2_omega(i_u_dot + m, i_grad_eta_f + n) = d2_omega(i_grad_eta_f + n, i_u_dot + m) = -F_inv[n][m] * c_f;
				}
				for(unsigned int i = 0; i < I; ++i)
				{
					for(unsigned int n = 0; n < 3; ++n)
					{
						d2_omega(i_v_dot + m, i_grad_eta_i[i] + n) = d2_omega(i_grad_eta_i[i] + n, i_v_dot + m) = F_inv[n][m] * c_i[i];
						d2_omega(i_u_dot + m, i_grad_eta_i[i] + n) = d2_omega(i_grad_eta_i[i] + n, i_u_dot + m) = -F_inv[n][m] * c_i[i];
					}
				}

			}


			if(compute_dq)
			{
				for(unsigned int m = 0; m < 3; ++m)
				{
					d2_omega(i_v_dot + m, i_c_f) = grad_eta_f_F_inv[m];
					d2_omega(i_u_dot + m, i_c_f) = -grad_eta_f_F_inv[m];
					d2_omega(i_grad_eta_f + m, i_c_f) = d_omega(i_grad_eta_f + m) / c_f;
					for(unsigned int i = 0; i < I; ++i)
					{
						d2_omega(i_v_dot + m, i_c_i[i]) = grad_eta_i_F_inv[i][m];
						d2_omega(i_u_dot + m, i_c_i[i]) = -grad_eta_i_F_inv[i][m];
						d2_omega(i_grad_eta_i[i] + m, i_c_i[i]) = d_omega(i_grad_eta_i[i] + m) / c_i[i];
 					}
					for(unsigned int k = 0; k < 3; ++k)
					{
						for(unsigned int l = 0; l < 3; ++l)
						{
							d2_omega(i_v_dot + m, i_F + 3 * k + l) = -grad_eta_c_f_F_inv[k] * F_inv[l][m];
							d2_omega(i_u_dot + m, i_F + 3 * k + l) = grad_eta_c_f_F_inv[k] * F_inv[l][m];
							d2_omega(i_grad_eta_f + m, i_F + 3 * k + l) = -F_inv_delta_v_dot[l] * F_inv[m][k] * c_f;
							for(unsigned int i = 0; i < I; ++i)
								d2_omega(i_grad_eta_i[i] + m, i_F + 3 * k + l) = -F_inv_delta_v_dot[l] * F_inv[m][k] * c_i[i];

						}
					}

				}

			}

		}
		return false;

	}
};

/**
 * Class defining an interface related scalar functional for an idealized description of electrolysis reactions with the integrand
 *
 * \f$ \omega^\Sigma =	-\begin{cases} \dfrac{1}{2R} \left[ \eta^\mathrm{int} - A^{\mathrm{e^-}} F \left(\bar{\varphi} - \varphi^\mathrm{c}\right) \right]^2 \quad &\mathrm{if} \quad \eta^\mathrm{int} - A^{\mathrm{e^-}}F\left(\bar{\varphi} - \varphi^\mathrm{c} \right) > 0\\ 0\quad &\mathrm{else}  \end{cases} \f$
 *
 * where \f$\eta^\mathrm{int}\f$ is the electrochemical potential in the interior of the domain,<br>
 * \f$F\f$ is Faraday's constant,<br>
 * \f$A^{\mathrm{e^-}}\f$ is the number of electrons added to the solution during the reaction,<br>
 * \f$\bar{\varphi}\f$ the prescribed external electrical potential,<br>
 * \f$\varphi^\mathrm{c}\f$ the overpotential required to start the reaction,<br>
 * and \f$R\f$ an "interface resistance"
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\eta^\mathrm{int}\f$
 */
template<unsigned int spacedim>
class OmegaElectrolysis00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * \f$F\f$
	 */
	const double
	F;

	/**
	 * \f$R\f$
	 */
	const double
	R;

	/**
	 * \f$A^{\mathrm{e^-}}\f$
	 */
	const double
	A_e;

	/**
	 * \f$\varphi^\mathrm{c}\f$
	 */
	const double
	phi_c;


	/**
	 * %Function determining \f$\bar{\varphi}(\boldsymbol{X}, t)\f$
	 */
	dealii::Function<spacedim>&
	phi_bar;

	/**
	 * This parameter is used to make the system definite in the range where the reaction does not take place.
	 * It represents a fraction of the second derivative of the potential in the range where the reaction does take place.
	 * This fraction is output as second derivative in the range where the reaction does not take place.
	 */
	const double
	regularization;

public:

	mutable	double
	delta_phi_lb = -DBL_MAX;

	mutable double
	delta_phi_ub = DBL_MAX;

	mutable double
	electrolysis_active = false;

	/**
	 * whether to take this scalar functional into account
	 */
	bool
	is_active = true;

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
	 * @param[in]		F						OmegaElectrolysis00::F
	 *
	 * @param[in]		R						OmegaElectrolysis00::R
	 *
	 * @param[in]		A_e						OmegaElectrolysis00::A_e
	 *
	 * @param[in]		phi_c					OmegaElectrolysis00::phi_c
	 *
	 * @param[in]		phi_bar					OmegaElectrolysis00::phi_bar
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 *
	 * @param[in]		regularization			OmegaElectrolysis00::regularization
	 */
	OmegaElectrolysis00(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						const double																	F,
						const double																	R,
						const double																	A_e,
						const double																	phi_c,
						dealii::Function<spacedim>&														phi_bar,
						const unsigned int																method,
						const double																	alpha = 0.0,
						const double																	regularization = 1e-12)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 0, 0, 1, 0, method, alpha, "OmegaElectrolysis00"),
	F(F),
	R(R),
	A_e(A_e),
	phi_c(phi_c),
	phi_bar(phi_bar),
	regularization(regularization)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{

		const double time_old = phi_bar.get_time();
		phi_bar.set_time(t);
		const double eta_bar_ = -A_e * F * phi_bar.value(x);
		phi_bar.set_time(time_old);
		const double eta_c = -A_e * F * phi_c;

		const double eta_int = values[0];
		const double delta_eta = eta_int + eta_bar_;

		if(A_e > 0)
		{
			const double delta_phi_ub_ = 1.0 / F / A_e * (eta_c - delta_eta);
			if(delta_phi_ub_ < delta_phi_ub)
				delta_phi_ub = delta_phi_ub_;
		}
		else
		{
			const double delta_phi_lb_ = 1.0 / F / A_e * (eta_c - delta_eta);
			if(delta_phi_lb_ > delta_phi_lb)
				delta_phi_lb = delta_phi_lb_;
		}

		if(delta_eta >= eta_c)
			electrolysis_active = true;

		if(is_active)
		{
			if(get<0>(requested_quantities))
			{
				if(delta_eta >= eta_c)
					sigma = -1.0 / (2.0 * R) * (delta_eta - eta_c) * (delta_eta - eta_c);
				else
					sigma = 0.0;
			}

			if(get<1>(requested_quantities))
			{
				if(delta_eta >= eta_c)
					d_sigma[0] = -1.0 / R * (delta_eta - eta_c);
				else
					d_sigma = 0.0;
			}

			if(get<2>(requested_quantities))
			{
				if(delta_eta >= eta_c)
					d2_sigma[0][0] = -1.0 / R;
				else
					// small second derivative to ensure definiteness of system
					d2_sigma[0][0] = -regularization / R;
			}
		}

		return false;
	}

};

/**
 * Class defining an interface related scalar functional for an idealized description of electrolysis reactions with the integrand
 *
 * \f$ \omega^\Sigma =	\inf\limits_{\dot{I}}[( \eta^\mathrm{int} - A^{\mathrm{e^-}} F \bar{\varphi} ) \dot{I} + \mathring{\delta}(\dot{I})] \f$,
 *
 * where \f$ \mathring{\delta} = -\dfrac{i_0 R T}{F} \left[ \sqrt{1+\left( \dfrac{F\dot{I}}{i_0} \right)^2} - 1 \right] + RT\dot{I}\mathrm{sinh}^{-1}\left(\dfrac{F\dot{I}}{i_0}\right) + \dfrac{R^\mathrm{el}}{2} \left( F \dot{I} \right)^2 \f$
 *
 * where \f$\eta^\mathrm{int}\f$ is the driving force for the reaction in the absence of an applied external potential,<br>
 * \f$F\f$ is Faraday's constant,<br>
 * \f$R\f$ is the gas constant,<br>
 * \f$T\f$ is the absolute temperature<br>
 * \f$A^{\mathrm{e^-}}\f$ is the number of electrons added to the solution during the reaction,<br>
 * \f$\bar{\varphi}\f$ the prescribed external electrical potential,<br>
 * \f$i_0\f$ the exchange current density,<br>
 * and \f$R^\mathrm{el}\f$ an electrical "interface resistance".
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\eta^\mathrm{int}\f$
 */
template<unsigned int spacedim>
class OmegaElectrolysis01 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * \f$F\f$
	 */
	const double
	F;

	/**
	 * \f$R T\f$
	 */
	const double
	RT;

	/**
	 * \f$A^{\mathrm{e^-}}\f$
	 */
	const double
	A_e;

	/**
	 * %Function determining \f$\bar{\varphi}(\boldsymbol{X}, t)\f$
	 */
	dealii::Function<spacedim>&
	phi_bar;

	/**
	 * \f$i_0\f$
	 */
	const double
	i_0;

	/**
	 * \f$R^\mathrm{el}\f$
	 */
	const double
	R_el;

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
	 * @param[in]		F						OmegaElectrolysis01::F
	 *
	 * @param[in]		RT						OmegaElectrolysis01::RT
	 *
	 * @param[in]		A_e						OmegaElectrolysis01::A_e
	 *
	 * @param[in]		phi_bar					OmegaElectrolysis01::phi_bar
	 *
	 * @param[in]		i_0						OmegaElectrolysis01::i_0
	 *
	 * @param[in]		R_el					OmegaElectrolysis01::R_el
	 *
	 * @param[in]		method					Omega::method
	 *
	 * @param[in]		alpha					Omega::alpha
	 *
	 */
	OmegaElectrolysis01(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
						const std::set<dealii::types::material_id>										domain_of_integration,
						const dealii::Quadrature<spacedim-1>											quadrature,
						GlobalDataIncrementalFE<spacedim>&												global_data,
						const double																	F,
						const double																	RT,
						const double																	A_e,
						dealii::Function<spacedim>&														phi_bar,
						const double																	i_0,
						const double																	R_el,
						const unsigned int																method,
						const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 0, 0, 1, 0, method, alpha, "OmegaElectrolysis01"),
	F(F),
	RT(RT),
	A_e(A_e),
	phi_bar(phi_bar),
	i_0(i_0),
	R_el(R_el)
	{
	}

	/**
	 * @see Omega::get_values_and_derivatives()
	 */
	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						t,
								const dealii::Point<spacedim>& 		x,
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							/*compute_dq*/)
	const
	{

		const double time_old = phi_bar.get_time();
		phi_bar.set_time(t);
		const double eta_bar_ = -A_e * F * phi_bar.value(x);
		phi_bar.set_time(time_old);

		const double eta_int = values[0];
		const double delta_eta = eta_int + eta_bar_;

		// iterate for corresponding current
		double I_k = 0.0;
		double f_old = 1e16;
		unsigned int iter = 0;
		for(;;)
		{
			iter++;
			const double f = asinh(I_k * F / i_0) + 1.0 / RT * R_el * F * F * I_k + delta_eta / RT;
			if(fabs(fabs(f) - fabs(f_old)) < 1e-16)
				break;
			if(iter == 100)
				return true;
			f_old = f;
			const double df_dI_k = (F / i_0) / sqrt( 1.0 + (I_k * F / i_0) * (I_k * F / i_0) ) + 1.0 / RT * R_el * F * F;
			const double dI_k = - f / df_dI_k;
			I_k = I_k + dI_k;
		}

		if(get<0>(requested_quantities))
			sigma = delta_eta * I_k - i_0 * RT / F * ( sqrt( 1.0 + (I_k * F / i_0) * (I_k * F / i_0) ) - 1.0 ) + RT * I_k * asinh(I_k * F / i_0) + 0.5 * R_el * (I_k * F) * (I_k * F);

		if(get<1>(requested_quantities))
			d_sigma[0] = I_k;

		if(get<2>(requested_quantities))
		{
			d2_sigma[0][0] = -1.0 / (RT * F / i_0 / sqrt( 1.0 + (I_k * F / i_0) * (I_k * F / i_0) ) + R_el * F * F);
		}

		return false;
	}

};

/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	\dfrac{A}{2n} (\boldsymbol{d}:\boldsymbol{d})^n \f$,
 *
 * where \f$\boldsymbol{d} = -\dfrac{1}{2}\left( \boldsymbol{Q}^{-1} \cdot \dot{\boldsymbol{Q}} + \dot{\boldsymbol{Q}} \cdot \boldsymbol{Q}^{-1} \right)\f$ is the stretching associated with the
 * symmetric right plastic stretch tensor \f$\boldsymbol{Q}^{-1}\f$, and \f$A\f$ and \f$n\f$ are material parameters.
 *
 * This describes an isotropic Norton type creep relation and can be used together with PsiElasticPlasticMaterial00 to model creep at large strains and small elastic deformations.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{Q}_{xx}\f$<br>
 * 																				[1]  \f$\dot{Q}_{xy}\f$<br>
 * 																				[2]  \f$\dot{Q}_{xz}\f$<br>
 * 																				[3]  \f$\dot{Q}_{yy}\f$<br>
 * 																				[4]  \f$\dot{Q}_{yz}\f$<br>
 * 																				[5]  \f$\dot{Q}_{zz}\f$<br>
 * 																				[6]  \f$Q_{xx}\f$<br>
 * 																				[7]  \f$Q_{xy}\f$<br>
 * 																				[8]  \f$Q_{xz}\f$<br>
 * 																				[9]  \f$Q_{yy}\f$<br>
 * 																				[10] \f$Q_{yz}\f$<br>
 * 																				[11] \f$Q_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaViscousDissipation00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * material parameter
	 */
	const double A;

	/**
	 * material parameter (creep exponent)
	 */
	const double n;

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
	 * @param[in]		A						OmegaViscousDissipation00::A
	 *
	 * @param[in]		n						OmegaViscousDissipation00::n
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaViscousDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																A,
								const double																n,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 0, 6, 0, 6, method, alpha, "OmegaViscousDissipation00"),
	A(A),
	n(n)
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

		dealii::Vector<double> Q_dot(6), Q(6), Q_inv(6);
		for(unsigned int m = 0; m < 6; ++m)
		{
			Q_dot[m] = values[m];
			Q[m] = values[m + 6];
		}
		dealii::SymmetricTensor<2,3> Q_t_inv;
		Q_t_inv[0][0] = Q[0];	Q_t_inv[0][1] = Q[1];	Q_t_inv[0][2] = Q[2];
								Q_t_inv[1][1] = Q[3];	Q_t_inv[1][2] = Q[4];
														Q_t_inv[2][2] = Q[5];
		Q_t_inv = invert(Q_t_inv);
		Q_inv[0] = Q_t_inv[0][0];	Q_inv[1] = Q_t_inv[0][1];	Q_inv[2] = Q_t_inv[0][2];
									Q_inv[3] = Q_t_inv[1][1];	Q_inv[4] = Q_t_inv[1][2];
																Q_inv[5] = Q_t_inv[2][2];

		dealii::Vector<double> d(9);
		d[0] = -Q_dot[0]*Q_inv[0] - Q_dot[1]*Q_inv[1] - Q_dot[2]*Q_inv[2];
		d[1] = -0.5*Q_dot[0]*Q_inv[1] - 0.5*Q_dot[1]*Q_inv[0] - 0.5*Q_dot[1]*Q_inv[3] - 0.5*Q_dot[2]*Q_inv[4] - 0.5*Q_dot[3]*Q_inv[1] - 0.5*Q_dot[4]*Q_inv[2];
		d[2] = -0.5*Q_dot[0]*Q_inv[2] - 0.5*Q_dot[1]*Q_inv[4] - 0.5*Q_dot[2]*Q_inv[0] - 0.5*Q_dot[2]*Q_inv[5] - 0.5*Q_dot[4]*Q_inv[1] - 0.5*Q_dot[5]*Q_inv[2];
		d[3] = -0.5*Q_dot[0]*Q_inv[1] - 0.5*Q_dot[1]*Q_inv[0] - 0.5*Q_dot[1]*Q_inv[3] - 0.5*Q_dot[2]*Q_inv[4] - 0.5*Q_dot[3]*Q_inv[1] - 0.5*Q_dot[4]*Q_inv[2];
		d[4] = -Q_dot[1]*Q_inv[1] - Q_dot[3]*Q_inv[3] - Q_dot[4]*Q_inv[4];
		d[5] = -0.5*Q_dot[1]*Q_inv[2] - 0.5*Q_dot[2]*Q_inv[1] - 0.5*Q_dot[3]*Q_inv[4] - 0.5*Q_dot[4]*Q_inv[3] - 0.5*Q_dot[4]*Q_inv[5] - 0.5*Q_dot[5]*Q_inv[4];
		d[6] = -0.5*Q_dot[0]*Q_inv[2] - 0.5*Q_dot[1]*Q_inv[4] - 0.5*Q_dot[2]*Q_inv[0] - 0.5*Q_dot[2]*Q_inv[5] - 0.5*Q_dot[4]*Q_inv[1] - 0.5*Q_dot[5]*Q_inv[2];
		d[7] = -0.5*Q_dot[1]*Q_inv[2] - 0.5*Q_dot[2]*Q_inv[1] - 0.5*Q_dot[3]*Q_inv[4] - 0.5*Q_dot[4]*Q_inv[3] - 0.5*Q_dot[4]*Q_inv[5] - 0.5*Q_dot[5]*Q_inv[4];
		d[8] = -Q_dot[2]*Q_inv[2] - Q_dot[4]*Q_inv[4] - Q_dot[5]*Q_inv[5];

		static dealii::FullMatrix<double> dd_dQ_dot(9,6), d2delta_dd2(9,9), d2delta_dd2_dd_dQ_dot(9,6), dd_dQ_dot_d2delta_dd2_dd_dQ_dot(6,6), dd_dQ_inv(9,6), d2delta_dd2_dd_dQ_dot_T_dd_dQ_inv(6,6), d_d2d_dQ_dot_d_Q_inv(6,6), d2delta_dQ_dot_dQ_inv(6,6), d2delta_dQ_dot_dQ(6,6), dB_dQ(6,6), dQ_inv_dQ(6,6);
		static dealii::Vector<double> B(6), ddetQ_dQ(6);
		if( (get<1>(requested_quantities)) || (get<2>(requested_quantities)))
		{
			dd_dQ_dot(0,0) = -Q_inv[0];		dd_dQ_dot(0,1) = -Q_inv[1];						dd_dQ_dot(0,2) = -Q_inv[2];						dd_dQ_dot(0,3) = 0;				dd_dQ_dot(0,4) = 0;								dd_dQ_dot(0,5) = 0;
			dd_dQ_dot(1,0) = -0.5*Q_inv[1];	dd_dQ_dot(1,1) = -0.5*Q_inv[0] - 0.5*Q_inv[3];	dd_dQ_dot(1,2) = -0.5*Q_inv[4];					dd_dQ_dot(1,3) = -0.5*Q_inv[1];	dd_dQ_dot(1,4) = -0.5*Q_inv[2];					dd_dQ_dot(1,5) = 0;
			dd_dQ_dot(2,0) = -0.5*Q_inv[2];	dd_dQ_dot(2,1) = -0.5*Q_inv[4];					dd_dQ_dot(2,2) = -0.5*Q_inv[0] - 0.5*Q_inv[5];	dd_dQ_dot(2,3) = 0;				dd_dQ_dot(2,4) = -0.5*Q_inv[1];					dd_dQ_dot(2,5) = -0.5*Q_inv[2];
			dd_dQ_dot(3,0) = -0.5*Q_inv[1];	dd_dQ_dot(3,1) = -0.5*Q_inv[0] - 0.5*Q_inv[3];	dd_dQ_dot(3,2) = -0.5*Q_inv[4];					dd_dQ_dot(3,3) = -0.5*Q_inv[1];	dd_dQ_dot(3,4) = -0.5*Q_inv[2];					dd_dQ_dot(3,5) = 0;
			dd_dQ_dot(4,0) = 0;				dd_dQ_dot(4,1) = -Q_inv[1];						dd_dQ_dot(4,2) = 0;								dd_dQ_dot(4,3) = -Q_inv[3];		dd_dQ_dot(4,4) = -Q_inv[4];						dd_dQ_dot(4,5) = 0;
			dd_dQ_dot(5,0) = 0;				dd_dQ_dot(5,1) = -0.5*Q_inv[2];					dd_dQ_dot(5,2) = -0.5*Q_inv[1];					dd_dQ_dot(5,3) = -0.5*Q_inv[4];	dd_dQ_dot(5,4) = -0.5*Q_inv[3] - 0.5*Q_inv[5];	dd_dQ_dot(5,5) = -0.5*Q_inv[4];
			dd_dQ_dot(6,0) = -0.5*Q_inv[2];	dd_dQ_dot(6,1) = -0.5*Q_inv[4];					dd_dQ_dot(6,2) = -0.5*Q_inv[0] - 0.5*Q_inv[5];	dd_dQ_dot(6,3) = 0;				dd_dQ_dot(6,4) = -0.5*Q_inv[1];					dd_dQ_dot(6,5) = -0.5*Q_inv[2];
			dd_dQ_dot(7,0) = 0;				dd_dQ_dot(7,1) = -0.5*Q_inv[2];					dd_dQ_dot(7,2) = -0.5*Q_inv[1];					dd_dQ_dot(7,3) = -0.5*Q_inv[4];	dd_dQ_dot(7,4) = -0.5*Q_inv[3] - 0.5*Q_inv[5];	dd_dQ_dot(7,5) = -0.5*Q_inv[4];
			dd_dQ_dot(8,0) = 0;				dd_dQ_dot(8,1) = 0;								dd_dQ_dot(8,2) = -Q_inv[2];						dd_dQ_dot(8,3) = 0;				dd_dQ_dot(8,4) = -Q_inv[4];						dd_dQ_dot(8,5) = -Q_inv[5];
		}

		if(get<2>(requested_quantities))
		{
			dd_dQ_inv(0,0) = -Q_dot[0];		dd_dQ_inv(0,1) = -Q_dot[1];						dd_dQ_inv(0,2) = -Q_dot[2];						dd_dQ_inv(0,3) = 0;				dd_dQ_inv(0,4) = 0;								dd_dQ_inv(0,5) = 0;
			dd_dQ_inv(1,0) = -0.5*Q_dot[1];	dd_dQ_inv(1,1) = -0.5*Q_dot[0] - 0.5*Q_dot[3];	dd_dQ_inv(1,2) = -0.5*Q_dot[4];					dd_dQ_inv(1,3) = -0.5*Q_dot[1];	dd_dQ_inv(1,4) = -0.5*Q_dot[2];					dd_dQ_inv(1,5) = 0;
			dd_dQ_inv(2,0) = -0.5*Q_dot[2];	dd_dQ_inv(2,1) = -0.5*Q_dot[4];					dd_dQ_inv(2,2) = -0.5*Q_dot[0] - 0.5*Q_dot[5];	dd_dQ_inv(2,3) = 0;				dd_dQ_inv(2,4) = -0.5*Q_dot[1];					dd_dQ_inv(2,5) = -0.5*Q_dot[2];
			dd_dQ_inv(3,0) = -0.5*Q_dot[1];	dd_dQ_inv(3,1) = -0.5*Q_dot[0] - 0.5*Q_dot[3];	dd_dQ_inv(3,2) = -0.5*Q_dot[4];					dd_dQ_inv(3,3) = -0.5*Q_dot[1];	dd_dQ_inv(3,4) = -0.5*Q_dot[2];					dd_dQ_inv(3,5) = 0;
			dd_dQ_inv(4,0) = 0;				dd_dQ_inv(4,1) = -Q_dot[1];						dd_dQ_inv(4,2) = 0;								dd_dQ_inv(4,3) = -Q_dot[3];		dd_dQ_inv(4,4) = -Q_dot[4];						dd_dQ_inv(4,5) = 0;
			dd_dQ_inv(5,0) = 0;				dd_dQ_inv(5,1) = -0.5*Q_dot[2];					dd_dQ_inv(5,2) = -0.5*Q_dot[1];					dd_dQ_inv(5,3) = -0.5*Q_dot[4];	dd_dQ_inv(5,4) = -0.5*Q_dot[3] - 0.5*Q_dot[5];	dd_dQ_inv(5,5) = -0.5*Q_dot[4];
			dd_dQ_inv(6,0) = -0.5*Q_dot[2];	dd_dQ_inv(6,1) = -0.5*Q_dot[4];					dd_dQ_inv(6,2) = -0.5*Q_dot[0] - 0.5*Q_dot[5];	dd_dQ_inv(6,3) = 0;				dd_dQ_inv(6,4) = -0.5*Q_dot[1];					dd_dQ_inv(6,5) = -0.5*Q_dot[2];
			dd_dQ_inv(7,0) = 0;				dd_dQ_inv(7,1) = -0.5*Q_dot[2];					dd_dQ_inv(7,2) = -0.5*Q_dot[1];					dd_dQ_inv(7,3) = -0.5*Q_dot[4];	dd_dQ_inv(7,4) = -0.5*Q_dot[3] - 0.5*Q_dot[5];	dd_dQ_inv(7,5) = -0.5*Q_dot[4];
			dd_dQ_inv(8,0) = 0;				dd_dQ_inv(8,1) = 0;								dd_dQ_inv(8,2) = -Q_dot[2];						dd_dQ_inv(8,3) = 0;				dd_dQ_inv(8,4) = -Q_dot[4];						dd_dQ_inv(8,5) = -Q_dot[5];

			d_d2d_dQ_dot_d_Q_inv(0,0) = -d[0];							d_d2d_dQ_dot_d_Q_inv(0,1) = -1.0/2.0*d[1] - 1.0/2.0*d[3];	d_d2d_dQ_dot_d_Q_inv(0,2) = -1.0/2.0*d[2] - 1.0/2.0*d[6];	d_d2d_dQ_dot_d_Q_inv(0,3) = 0;								d_d2d_dQ_dot_d_Q_inv(0,4) = 0;								d_d2d_dQ_dot_d_Q_inv(0,5) = 0;
			d_d2d_dQ_dot_d_Q_inv(1,0) = -1.0/2.0*d[1] - 1.0/2.0*d[3];	d_d2d_dQ_dot_d_Q_inv(1,1) = -d[0] - d[4];					d_d2d_dQ_dot_d_Q_inv(1,2) = -1.0/2.0*d[5] - 1.0/2.0*d[7];	d_d2d_dQ_dot_d_Q_inv(1,3) = -1.0/2.0*d[1] - 1.0/2.0*d[3];	d_d2d_dQ_dot_d_Q_inv(1,4) = -1.0/2.0*d[2] - 1.0/2.0*d[6];	d_d2d_dQ_dot_d_Q_inv(1,5) = 0;
			d_d2d_dQ_dot_d_Q_inv(2,0) = -1.0/2.0*d[2] - 1.0/2.0*d[6];	d_d2d_dQ_dot_d_Q_inv(2,1) = -1.0/2.0*d[5] - 1.0/2.0*d[7];	d_d2d_dQ_dot_d_Q_inv(2,2) = -d[0] - d[8];					d_d2d_dQ_dot_d_Q_inv(2,3) = 0;								d_d2d_dQ_dot_d_Q_inv(2,4) = -1.0/2.0*d[1] - 1.0/2.0*d[3];	d_d2d_dQ_dot_d_Q_inv(2,5) = -1.0/2.0*d[2] - 1.0/2.0*d[6];
			d_d2d_dQ_dot_d_Q_inv(3,0) = 0;								d_d2d_dQ_dot_d_Q_inv(3,1) = -1.0/2.0*d[1] - 1.0/2.0*d[3];	d_d2d_dQ_dot_d_Q_inv(3,2) = 0;								d_d2d_dQ_dot_d_Q_inv(3,3) = -d[4];							d_d2d_dQ_dot_d_Q_inv(3,4) = -1.0/2.0*d[5] - 1.0/2.0*d[7];	d_d2d_dQ_dot_d_Q_inv(3,5) = 0;
			d_d2d_dQ_dot_d_Q_inv(4,0) = 0;								d_d2d_dQ_dot_d_Q_inv(4,1) = -1.0/2.0*d[2] - 1.0/2.0*d[6];	d_d2d_dQ_dot_d_Q_inv(4,2) = -1.0/2.0*d[1] - 1.0/2.0*d[3];	d_d2d_dQ_dot_d_Q_inv(4,3) = -1.0/2.0*d[5] - 1.0/2.0*d[7];	d_d2d_dQ_dot_d_Q_inv(4,4) = -d[4] - d[8];					d_d2d_dQ_dot_d_Q_inv(4,5) = -1.0/2.0*d[5] - 1.0/2.0*d[7];
			d_d2d_dQ_dot_d_Q_inv(5,0) = 0;								d_d2d_dQ_dot_d_Q_inv(5,1) = 0;								d_d2d_dQ_dot_d_Q_inv(5,2) = -1.0/2.0*d[2] - 1.0/2.0*d[6];	d_d2d_dQ_dot_d_Q_inv(5,3) = 0;								d_d2d_dQ_dot_d_Q_inv(5,4) = -1.0/2.0*d[5] - 1.0/2.0*d[7];	d_d2d_dQ_dot_d_Q_inv(5,5) = -d[8];

			B[0] =  Q[3]*Q[5] - Q[4]*Q[4];
			B[1] = -Q[1]*Q[5] + Q[2]*Q[4];
			B[2] =  Q[1]*Q[4] - Q[2]*Q[3];
			B[3] =  Q[0]*Q[5] - Q[2]*Q[2];
			B[4] = -Q[0]*Q[4] + Q[1]*Q[2];
			B[5] =  Q[0]*Q[3] - Q[1]*Q[1];

			dB_dQ(0,0) = 0;		dB_dQ(0,1) = 0;			dB_dQ(0,2) = 0;			dB_dQ(0,3) = Q[5];		dB_dQ(0,4) = -2.0*Q[4];	dB_dQ(0,5) = Q[3];
			dB_dQ(1,0) = 0;		dB_dQ(1,1) = -Q[5];		dB_dQ(1,2) = Q[4];		dB_dQ(1,3) = 0;			dB_dQ(1,4) = Q[2];		dB_dQ(1,5) = -Q[1];
			dB_dQ(2,0) = 0;		dB_dQ(2,1) = Q[4];		dB_dQ(2,2) = -Q[3];		dB_dQ(2,3) = -Q[2];		dB_dQ(2,4) = Q[1];		dB_dQ(2,5) = 0;
			dB_dQ(3,0) = Q[5];	dB_dQ(3,1) = 0;			dB_dQ(3,2) = -2.0*Q[2];	dB_dQ(3,3) = 0;			dB_dQ(3,4) = 0;			dB_dQ(3,5) = Q[0];
			dB_dQ(4,0) = -Q[4];	dB_dQ(4,1) = Q[2];		dB_dQ(4,2) = Q[1];		dB_dQ(4,3) = 0;			dB_dQ(4,4) = -Q[0];		dB_dQ(4,5) = 0;
			dB_dQ(5,0) = Q[3];	dB_dQ(5,1) = -2.0*Q[1];	dB_dQ(5,2) = 0;			dB_dQ(5,3) = Q[0];		dB_dQ(5,4) = 0;			dB_dQ(5,5) = 0;

			const double detQ = get_J(Q, true);
			get_dJ_dF(Q, ddetQ_dQ, true);
			for(unsigned int i = 0; i < 6; ++i)
				for(unsigned int j = 0; j < 6; ++j)
					dQ_inv_dQ(i,j) = ( dB_dQ(i,j) - B[i] * ddetQ_dQ[j] / detQ ) / detQ;

		}

		const double dd = d * d;
		const double dd_n = dd > 0.0 ? pow(dd, n) : 0.0;
		const double dd_n_1 = dd > 0.0 ? pow(dd, n-1.0) : 0.0;
		const double dd_n_2 = dd > 0.0 ? pow(dd, n-2.0) : 0.0;

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * A/n * dd_n;
		}

		if(get<1>(requested_quantities))
		{
			dealii::Vector<double> d_dd_dQ_dot(6);
			dd_dQ_dot.Tvmult(d_dd_dQ_dot, d);
			for(unsigned int m = 0; m < 6; ++m)
				d_omega[m] = A * dd_n_1 * d_dd_dQ_dot[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int i = 0; i < 9; ++i)
			{
				for(unsigned int j = 0; j < 9; ++j)
					d2delta_dd2(i,j) = 2.0 * A * (n - 1.0) * dd_n_2 * d[i] * d[j];
				d2delta_dd2(i,i) += A * dd_n_1;
			}
			d2delta_dd2.mmult(d2delta_dd2_dd_dQ_dot, dd_dQ_dot);
			dd_dQ_dot.Tmmult(dd_dQ_dot_d2delta_dd2_dd_dQ_dot, d2delta_dd2_dd_dQ_dot);
			for(unsigned int i = 0; i < 6; ++i)
			{
				for(unsigned int j = 0; j < 6; ++j)
					d2_omega(i,j) = dd_dQ_dot_d2delta_dd2_dd_dQ_dot(i,j);
			}

			if(compute_dq)
			{
				d2delta_dd2_dd_dQ_dot.Tmmult(d2delta_dd2_dd_dQ_dot_T_dd_dQ_inv, dd_dQ_inv);
				for(unsigned int i = 0; i < 6; ++i)
					for(unsigned int j = 0; j < 6; ++j)
						d2delta_dQ_dot_dQ_inv(i,j) = d2delta_dd2_dd_dQ_dot_T_dd_dQ_inv(i,j) + A * dd_n_1 * d_d2d_dQ_dot_d_Q_inv(i,j);
				d2delta_dQ_dot_dQ_inv.mmult(d2delta_dQ_dot_dQ, dQ_inv_dQ);
				for(unsigned int i = 0; i < 6; ++i)
					for(unsigned int j = 0; j < 6; ++j)
						d2_omega(i,j + 6) = d2delta_dQ_dot_dQ(i,j);
			}

		}

		return false;
	}

};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	\dfrac{A}{2n} (\boldsymbol{d}:\boldsymbol{d})^n \f$,
 *
 * where \f$\boldsymbol{d} = \dfrac{1}{2}\left( \boldsymbol{F}^{-1} \cdot \dot{\boldsymbol{F}} + \dot{\boldsymbol{F}} \cdot \boldsymbol{F}^{-1} \right)\f$ is the stretching associated with the
 * deformation gradient \f$\boldsymbol{F}\f$, and \f$A\f$ and \f$n\f$ are material parameters.
 *
 * In order to avoid singular second derivatives,
 *
 * \f$h^\Omega_\rho =	\dfrac{A_0}{2} \boldsymbol{d}:\boldsymbol{d}\f$
 *
 * is used if \f$\sqrt{\boldsymbol{d}:\boldsymbol{d}} < d_\mathrm{th}\f$, with \f$d_\mathrm{th}\ll 1\f$ and \f$A_0 = A d_\mathrm{th}^{2(n-1)} \f$. In practice, \f$d_\mathrm{th}\f$ should
 * be chosen such that it is slightly smaller than the maximum of \f$\sqrt{\boldsymbol{d}:\boldsymbol{d}}\f$ in the problem. If \f$n > 1\f$, \f$d_\mathrm{th}\f$ should be set to zero.
 *
 * This describes an isotropic Norton type creep relation at large strains.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{F}_{xx}\f$<br>
 * 																				[1]  \f$\dot{F}_{xy}\f$<br>
 * 																				[2]  \f$\dot{F}_{xz}\f$<br>
 * 																				[3]  \f$\dot{F}_{yx}\f$<br>
 * 																				[4]  \f$\dot{F}_{yy}\f$<br>
 * 																				[5]  \f$\dot{F}_{yz}\f$<br>
 * 																				[6]  \f$\dot{F}_{zx}\f$<br>
 * 																				[7]  \f$\dot{F}_{zy}\f$<br>
 * 																				[8]  \f$\dot{F}_{zz}\f$<br>
 * 																				[9]  \f$F_{xx}\f$<br>
 * 																				[10] \f$F_{xy}\f$<br>
 * 																				[11] \f$F_{xz}\f$<br>
 * 																				[12] \f$F_{yx}\f$<br>
 * 																				[13] \f$F_{yy}\f$<br>
 * 																				[14] \f$F_{yz}\f$<br>
 * 																				[15] \f$F_{zx}\f$<br>
 * 																				[16] \f$F_{zy}\f$<br>
 * 																				[17] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaViscousDissipation01 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * material parameter
	 */
	const double
	A;

	/**
	 * material parameter (creep exponent)
	 */
	const double
	n;

	/**
	 * Uniaxial threshold stretching below which potential is quadratically approximated with regard to the stretching
	 */
	const double
	d_th = 1e-8;

	/**
	 * global data object
	 */
	GlobalDataIncrementalFE<spacedim>&
	global_data;

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
	 * @param[in]		A						OmegaViscousDissipation01::A
	 *
	 * @param[in]		n						OmegaViscousDissipation01::n
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 *
	 * @param[in]		d_th					OmegaViscousDissipation01::d_th
	 */
	OmegaViscousDissipation01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																A,
								const double																n,
								const unsigned int															method,
								const double																alpha = 0.0,
								const double																d_th = 1e-8)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 0, 9, 0, 9, method, alpha, "OmegaViscousDissipation01"),
	A(A),
	n(n),
	d_th(d_th),
	global_data(global_data)
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

		dealii::Vector<double> F_dot(9), F(9), F_inv(9);
		for(unsigned int m = 0; m < 9; ++m)
		{
			F_dot[m] = values[m];
			F[m] = values[m + 9];
		}
		dealii::Tensor<2,3> F_t_inv;
		F_t_inv[0][0] = F[0];	F_t_inv[0][1] = F[1];	F_t_inv[0][2] = F[2];
		F_t_inv[1][0] = F[3];	F_t_inv[1][1] = F[4];	F_t_inv[1][2] = F[5];
		F_t_inv[2][0] = F[6];	F_t_inv[2][1] = F[7];	F_t_inv[2][2] = F[8];
		F_t_inv = invert(F_t_inv);
		F_inv[0] = F_t_inv[0][0];	F_inv[1] = F_t_inv[0][1];	F_inv[2] = F_t_inv[0][2];
		F_inv[3] = F_t_inv[1][0];	F_inv[4] = F_t_inv[1][1];	F_inv[5] = F_t_inv[1][2];
		F_inv[6] = F_t_inv[2][0];	F_inv[7] = F_t_inv[2][1];	F_inv[8] = F_t_inv[2][2];

		dealii::Vector<double> d(9);
		d[0] = F_dot[0]*F_inv[0] + 0.5*F_dot[1]*F_inv[3] + 0.5*F_dot[2]*F_inv[6] + 0.5*F_dot[3]*F_inv[1] + 0.5*F_dot[6]*F_inv[2];
		d[1] = 0.5*F_dot[0]*F_inv[1] + 0.5*F_dot[1]*F_inv[0] + 0.5*F_dot[1]*F_inv[4] + 0.5*F_dot[2]*F_inv[7] + 0.5*F_dot[4]*F_inv[1] + 0.5*F_dot[7]*F_inv[2];
		d[2] = 0.5*F_dot[0]*F_inv[2] + 0.5*F_dot[1]*F_inv[5] + 0.5*F_dot[2]*F_inv[0] + 0.5*F_dot[2]*F_inv[8] + 0.5*F_dot[5]*F_inv[1] + 0.5*F_dot[8]*F_inv[2];
		d[3] = 0.5*F_dot[0]*F_inv[3] + 0.5*F_dot[3]*F_inv[0] + 0.5*F_dot[3]*F_inv[4] + 0.5*F_dot[4]*F_inv[3] + 0.5*F_dot[5]*F_inv[6] + 0.5*F_dot[6]*F_inv[5];
		d[4] = 0.5*F_dot[1]*F_inv[3] + 0.5*F_dot[3]*F_inv[1] + F_dot[4]*F_inv[4] + 0.5*F_dot[5]*F_inv[7] + 0.5*F_dot[7]*F_inv[5];
		d[5] = 0.5*F_dot[2]*F_inv[3] + 0.5*F_dot[3]*F_inv[2] + 0.5*F_dot[4]*F_inv[5] + 0.5*F_dot[5]*F_inv[4] + 0.5*F_dot[5]*F_inv[8] + 0.5*F_dot[8]*F_inv[5];
		d[6] = 0.5*F_dot[0]*F_inv[6] + 0.5*F_dot[3]*F_inv[7] + 0.5*F_dot[6]*F_inv[0] + 0.5*F_dot[6]*F_inv[8] + 0.5*F_dot[7]*F_inv[3] + 0.5*F_dot[8]*F_inv[6];
		d[7] = 0.5*F_dot[1]*F_inv[6] + 0.5*F_dot[4]*F_inv[7] + 0.5*F_dot[6]*F_inv[1] + 0.5*F_dot[7]*F_inv[4] + 0.5*F_dot[7]*F_inv[8] + 0.5*F_dot[8]*F_inv[7];
		d[8] = 0.5*F_dot[2]*F_inv[6] + 0.5*F_dot[5]*F_inv[7] + 0.5*F_dot[6]*F_inv[2] + 0.5*F_dot[7]*F_inv[5] + F_dot[8]*F_inv[8];

		static dealii::Vector<double> ddetF_dF(9), d_dd_dF_dot(9), d_dd_dF_inv(9);
		static dealii::FullMatrix<double> dd_dF_dot(9,9), dd_dF_inv(9,9), d_d2d_dF_dot_d_F_inv(9,9), d2delta_dF_dot2(9,9), d2delta_dF_dot_dF_inv(9,9), d2delta_dF_dot_dF(9,9), dF_inv_dF(9,9);
		if( (get<1>(requested_quantities)) || (get<2>(requested_quantities)))
		{
			dd_dF_dot(0,0) = F_inv[0];		dd_dF_dot(0,1) = 0.5*F_inv[3];					dd_dF_dot(0,2) = 0.5*F_inv[6];					dd_dF_dot(0,3) = 0.5*F_inv[1];					dd_dF_dot(0,4) = 0;				dd_dF_dot(0,5) = 0;								dd_dF_dot(0,6) = 0.5*F_inv[2];					dd_dF_dot(0,7) = 0;								dd_dF_dot(0,8) = 0;
			dd_dF_dot(1,0) = 0.5*F_inv[1];	dd_dF_dot(1,1) = 0.5*F_inv[0] + 0.5*F_inv[4];	dd_dF_dot(1,2) = 0.5*F_inv[7];					dd_dF_dot(1,3) = 0;								dd_dF_dot(1,4) = 0.5*F_inv[1];	dd_dF_dot(1,5) = 0;								dd_dF_dot(1,6) = 0;								dd_dF_dot(1,7) = 0.5*F_inv[2];					dd_dF_dot(1,8) = 0;
			dd_dF_dot(2,0) = 0.5*F_inv[2];	dd_dF_dot(2,1) = 0.5*F_inv[5];					dd_dF_dot(2,2) = 0.5*F_inv[0] + 0.5*F_inv[8];	dd_dF_dot(2,3) = 0;								dd_dF_dot(2,4) = 0;				dd_dF_dot(2,5) = 0.5*F_inv[1];					dd_dF_dot(2,6) = 0;								dd_dF_dot(2,7) = 0;								dd_dF_dot(2,8) = 0.5*F_inv[2];
			dd_dF_dot(3,0) = 0.5*F_inv[3];	dd_dF_dot(3,1) = 0;								dd_dF_dot(3,2) = 0;								dd_dF_dot(3,3) = 0.5*F_inv[0] + 0.5*F_inv[4];	dd_dF_dot(3,4) = 0.5*F_inv[3];	dd_dF_dot(3,5) = 0.5*F_inv[6];					dd_dF_dot(3,6) = 0.5*F_inv[5];					dd_dF_dot(3,7) = 0;								dd_dF_dot(3,8) = 0;
			dd_dF_dot(4,0) = 0;				dd_dF_dot(4,1) = 0.5*F_inv[3];					dd_dF_dot(4,2) = 0;								dd_dF_dot(4,3) = 0.5*F_inv[1];					dd_dF_dot(4,4) = F_inv[4];		dd_dF_dot(4,5) = 0.5*F_inv[7];					dd_dF_dot(4,6) = 0;								dd_dF_dot(4,7) = 0.5*F_inv[5];					dd_dF_dot(4,8) = 0;
			dd_dF_dot(5,0) = 0;				dd_dF_dot(5,1) = 0;								dd_dF_dot(5,2) = 0.5*F_inv[3];					dd_dF_dot(5,3) = 0.5*F_inv[2];					dd_dF_dot(5,4) = 0.5*F_inv[5];	dd_dF_dot(5,5) = 0.5*F_inv[4] + 0.5*F_inv[8];	dd_dF_dot(5,6) = 0;								dd_dF_dot(5,7) = 0;								dd_dF_dot(5,8) = 0.5*F_inv[5];
			dd_dF_dot(6,0) = 0.5*F_inv[6];	dd_dF_dot(6,1) = 0;								dd_dF_dot(6,2) = 0;								dd_dF_dot(6,3) = 0.5*F_inv[7];					dd_dF_dot(6,4) = 0;				dd_dF_dot(6,5) = 0;								dd_dF_dot(6,6) = 0.5*F_inv[0] + 0.5*F_inv[8];	dd_dF_dot(6,7) = 0.5*F_inv[3];					dd_dF_dot(6,8) = 0.5*F_inv[6];
			dd_dF_dot(7,0) = 0;				dd_dF_dot(7,1) = 0.5*F_inv[6];					dd_dF_dot(7,2) = 0;								dd_dF_dot(7,3) = 0;								dd_dF_dot(7,4) = 0.5*F_inv[7];	dd_dF_dot(7,5) = 0;								dd_dF_dot(7,6) = 0.5*F_inv[1];					dd_dF_dot(7,7) = 0.5*F_inv[4] + 0.5*F_inv[8];	dd_dF_dot(7,8) = 0.5*F_inv[7];
			dd_dF_dot(8,0) = 0;				dd_dF_dot(8,1) = 0;								dd_dF_dot(8,2) = 0.5*F_inv[6];					dd_dF_dot(8,3) = 0;								dd_dF_dot(8,4) = 0;				dd_dF_dot(8,5) = 0.5*F_inv[7];					dd_dF_dot(8,6) = 0.5*F_inv[2];					dd_dF_dot(8,7) = 0.5*F_inv[5];					dd_dF_dot(8,8) = F_inv[8];

			dd_dF_dot.Tvmult(d_dd_dF_dot, d);
		}

		const double dd = d * d;
		const double sqrt_dd = sqrt(dd);
		const double A_0 = A * pow(d_th, 2.0 * (n - 1.0));
		const double h = sqrt_dd > d_th ? A / n * pow(dd, n) : A_0 * dd;
		const double dh = sqrt_dd > d_th ? A * pow(dd, n - 1.0) : A_0;
		const double d2h = sqrt_dd > d_th ? A * (n - 1.0) * pow(dd, n - 2.0) : 0.0;

//		if(sqrt_dd < d_th)
//			global_data.set_not_converged_at_local_level();

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * h;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = dh * d_dd_dF_dot[m];
		}

		if(get<2>(requested_quantities))
		{
			dd_dF_dot.Tmmult(d2delta_dF_dot2, dd_dF_dot);
			d2delta_dF_dot2 *= dh;
			for(unsigned int i = 0; i < 9; ++i)
				for(unsigned int j = 0; j < 9; ++j)
					d2delta_dF_dot2(i,j) += 2.0 * d2h * d_dd_dF_dot[i] * d_dd_dF_dot[j];

			for(unsigned int i = 0; i < 9; ++i)
				for(unsigned int j = 0; j < 9; ++j)
					d2_omega(i,j) = d2delta_dF_dot2(i,j);


			if(compute_dq)
			{
				dd_dF_inv(0,0) = F_dot[0];		dd_dF_inv(0,1) = 0.5*F_dot[3];					dd_dF_inv(0,2) = 0.5*F_dot[6];					dd_dF_inv(0,3) = 0.5*F_dot[1];					dd_dF_inv(0,4) = 0;				dd_dF_inv(0,5) = 0;								dd_dF_inv(0,6) = 0.5*F_dot[2];					dd_dF_inv(0,7) = 0;								dd_dF_inv(0,8) = 0;
				dd_dF_inv(1,0) = 0.5*F_dot[1];	dd_dF_inv(1,1) = 0.5*F_dot[0] + 0.5*F_dot[4];	dd_dF_inv(1,2) = 0.5*F_dot[7];					dd_dF_inv(1,3) = 0;								dd_dF_inv(1,4) = 0.5*F_dot[1];	dd_dF_inv(1,5) = 0;								dd_dF_inv(1,6) = 0;								dd_dF_inv(1,7) = 0.5*F_dot[2];					dd_dF_inv(1,8) = 0;
				dd_dF_inv(2,0) = 0.5*F_dot[2];	dd_dF_inv(2,1) = 0.5*F_dot[5];					dd_dF_inv(2,2) = 0.5*F_dot[0] + 0.5*F_dot[8];	dd_dF_inv(2,3) = 0;								dd_dF_inv(2,4) = 0;				dd_dF_inv(2,5) = 0.5*F_dot[1];					dd_dF_inv(2,6) = 0;								dd_dF_inv(2,7) = 0;								dd_dF_inv(2,8) = 0.5*F_dot[2];
				dd_dF_inv(3,0) = 0.5*F_dot[3];	dd_dF_inv(3,1) = 0;								dd_dF_inv(3,2) = 0;								dd_dF_inv(3,3) = 0.5*F_dot[0] + 0.5*F_dot[4];	dd_dF_inv(3,4) = 0.5*F_dot[3];	dd_dF_inv(3,5) = 0.5*F_dot[6];					dd_dF_inv(3,6) = 0.5*F_dot[5];					dd_dF_inv(3,7) = 0;								dd_dF_inv(3,8) = 0;
				dd_dF_inv(4,0) = 0;				dd_dF_inv(4,1) = 0.5*F_dot[3];					dd_dF_inv(4,2) = 0;								dd_dF_inv(4,3) = 0.5*F_dot[1];					dd_dF_inv(4,4) = F_dot[4];		dd_dF_inv(4,5) = 0.5*F_dot[7];					dd_dF_inv(4,6) = 0;								dd_dF_inv(4,7) = 0.5*F_dot[5];					dd_dF_inv(4,8) = 0;
				dd_dF_inv(5,0) = 0;				dd_dF_inv(5,1) = 0;								dd_dF_inv(5,2) = 0.5*F_dot[3];					dd_dF_inv(5,3) = 0.5*F_dot[2];					dd_dF_inv(5,4) = 0.5*F_dot[5];	dd_dF_inv(5,5) = 0.5*F_dot[4] + 0.5*F_dot[8];	dd_dF_inv(5,6) = 0;								dd_dF_inv(5,7) = 0;								dd_dF_inv(5,8) = 0.5*F_dot[5];
				dd_dF_inv(6,0) = 0.5*F_dot[6];	dd_dF_inv(6,1) = 0;								dd_dF_inv(6,2) = 0;								dd_dF_inv(6,3) = 0.5*F_dot[7];					dd_dF_inv(6,4) = 0;				dd_dF_inv(6,5) = 0;								dd_dF_inv(6,6) = 0.5*F_dot[0] + 0.5*F_dot[8];	dd_dF_inv(6,7) = 0.5*F_dot[3];					dd_dF_inv(6,8) = 0.5*F_dot[6];
				dd_dF_inv(7,0) = 0;				dd_dF_inv(7,1) = 0.5*F_dot[6];					dd_dF_inv(7,2) = 0;								dd_dF_inv(7,3) = 0;								dd_dF_inv(7,4) = 0.5*F_dot[7];	dd_dF_inv(7,5) = 0;								dd_dF_inv(7,6) = 0.5*F_dot[1];					dd_dF_inv(7,7) = 0.5*F_dot[4] + 0.5*F_dot[8];	dd_dF_inv(7,8) = 0.5*F_dot[7];
				dd_dF_inv(8,0) = 0;				dd_dF_inv(8,1) = 0;								dd_dF_inv(8,2) = 0.5*F_dot[6];					dd_dF_inv(8,3) = 0;								dd_dF_inv(8,4) = 0;				dd_dF_inv(8,5) = 0.5*F_dot[7];					dd_dF_inv(8,6) = 0.5*F_dot[2];					dd_dF_inv(8,7) = 0.5*F_dot[5];					dd_dF_inv(8,8) = F_dot[8];

				d_d2d_dF_dot_d_F_inv(0,0) = d[0];				d_d2d_dF_dot_d_F_inv(0,1) = 0.5*d[1];			d_d2d_dF_dot_d_F_inv(0,2) = 0.5*d[2];				d_d2d_dF_dot_d_F_inv(0,3) = 0.5*d[3];			d_d2d_dF_dot_d_F_inv(0,4) = 0;					d_d2d_dF_dot_d_F_inv(0,5) = 0;					d_d2d_dF_dot_d_F_inv(0,6) = 0.5*d[6];			d_d2d_dF_dot_d_F_inv(0,7) = 0;					d_d2d_dF_dot_d_F_inv(0,8) = 0;
				d_d2d_dF_dot_d_F_inv(1,0) = 0.5*d[1];	 		d_d2d_dF_dot_d_F_inv(1,1) = 0;					d_d2d_dF_dot_d_F_inv(1,2) = 0;						d_d2d_dF_dot_d_F_inv(1,3) = 0.5*d[0] + 0.5*d[4];d_d2d_dF_dot_d_F_inv(1,4) = 0.5*d[1];			d_d2d_dF_dot_d_F_inv(1,5) = 0.5*d[2];			d_d2d_dF_dot_d_F_inv(1,6) = 0.5*d[7];			d_d2d_dF_dot_d_F_inv(1,7) = 0;					d_d2d_dF_dot_d_F_inv(1,8) = 0;
				d_d2d_dF_dot_d_F_inv(2,0) = 0.5*d[2];			d_d2d_dF_dot_d_F_inv(2,1) = 0;					d_d2d_dF_dot_d_F_inv(2,2) = 0;						d_d2d_dF_dot_d_F_inv(2,3) = 0.5*d[5];			d_d2d_dF_dot_d_F_inv(2,4) = 0;					d_d2d_dF_dot_d_F_inv(2,5) = 0;					d_d2d_dF_dot_d_F_inv(2,6) = 0.5*d[0] + 0.5*d[8];d_d2d_dF_dot_d_F_inv(2,7) = 0.5*d[1];			d_d2d_dF_dot_d_F_inv(2,8) = 0.5*d[2];
				d_d2d_dF_dot_d_F_inv(3,0) = 0.5*d[3];			d_d2d_dF_dot_d_F_inv(3,1) = 0.5*d[0] + 0.5*d[4];d_d2d_dF_dot_d_F_inv(3,2) = 0.5*d[5];				d_d2d_dF_dot_d_F_inv(3,3) = 0;					d_d2d_dF_dot_d_F_inv(3,4) = 0.5*d[3];			d_d2d_dF_dot_d_F_inv(3,5) = 0;					d_d2d_dF_dot_d_F_inv(3,6) = 0;					d_d2d_dF_dot_d_F_inv(3,7) = 0.5*d[6];			d_d2d_dF_dot_d_F_inv(3,8) = 0;
				d_d2d_dF_dot_d_F_inv(4,0) = 0;					d_d2d_dF_dot_d_F_inv(4,1) = 0.5*d[1];			d_d2d_dF_dot_d_F_inv(4,2) = 0;						d_d2d_dF_dot_d_F_inv(4,3) = 0.5*d[3];			d_d2d_dF_dot_d_F_inv(4,4) = d[4];				d_d2d_dF_dot_d_F_inv(4,5) = 0.5*d[5];			d_d2d_dF_dot_d_F_inv(4,6) = 0;					d_d2d_dF_dot_d_F_inv(4,7) = 0.5*d[7];			d_d2d_dF_dot_d_F_inv(4,8) = 0;
				d_d2d_dF_dot_d_F_inv(5,0) = 0;					d_d2d_dF_dot_d_F_inv(5,1) = 0.5*d[2];			d_d2d_dF_dot_d_F_inv(5,2) = 0;						d_d2d_dF_dot_d_F_inv(5,3) = 0;					d_d2d_dF_dot_d_F_inv(5,4) = 0.5*d[5];			d_d2d_dF_dot_d_F_inv(5,5) = 0;					d_d2d_dF_dot_d_F_inv(5,6) = 0.5*d[3];			d_d2d_dF_dot_d_F_inv(5,7) = 0.5*d[4] + 0.5*d[8];d_d2d_dF_dot_d_F_inv(5,8) = 0.5*d[5];
				d_d2d_dF_dot_d_F_inv(6,0) = 0.5*d[6];			d_d2d_dF_dot_d_F_inv(6,1) = 0.5*d[7];			d_d2d_dF_dot_d_F_inv(6,2) = 0.5*d[0] + 0.5*d[8];	d_d2d_dF_dot_d_F_inv(6,3) = 0;					d_d2d_dF_dot_d_F_inv(6,4) = 0;					d_d2d_dF_dot_d_F_inv(6,5) = 0.5*d[3];			d_d2d_dF_dot_d_F_inv(6,6) = 0;					d_d2d_dF_dot_d_F_inv(6,7) = 0;					d_d2d_dF_dot_d_F_inv(6,8) = 0.5*d[6];
				d_d2d_dF_dot_d_F_inv(7,0) = 0;					d_d2d_dF_dot_d_F_inv(7,1) = 0;					d_d2d_dF_dot_d_F_inv(7,2) = 0.5*d[1];				d_d2d_dF_dot_d_F_inv(7,3) = 0.5*d[6];			d_d2d_dF_dot_d_F_inv(7,4) = 0.5*d[7];			d_d2d_dF_dot_d_F_inv(7,5) = 0.5*d[4] + 0.5*d[8];d_d2d_dF_dot_d_F_inv(7,6) = 0;					d_d2d_dF_dot_d_F_inv(7,7) = 0;					d_d2d_dF_dot_d_F_inv(7,8) = 0.5*d[7];
				d_d2d_dF_dot_d_F_inv(8,0) = 0;					d_d2d_dF_dot_d_F_inv(8,1) = 0;					d_d2d_dF_dot_d_F_inv(8,2) = 0.5*d[2];				d_d2d_dF_dot_d_F_inv(8,3) = 0;					d_d2d_dF_dot_d_F_inv(8,4) = 0;					d_d2d_dF_dot_d_F_inv(8,5) = 0.5*d[5];			d_d2d_dF_dot_d_F_inv(8,6) = 0.5*d[6];			d_d2d_dF_dot_d_F_inv(8,7) = 0.5*d[7];			d_d2d_dF_dot_d_F_inv(8,8) = d[8];

				dF_inv_dF(0,0) = -F[4]*F[4]*F[8]*F[8] + 2.0*F[4]*F[5]*F[7]*F[8] - F[5]*F[5]*F[7]*F[7];
				dF_inv_dF(0,1) = F[3]*F[4]*F[8]*F[8] - F[3]*F[5]*F[7]*F[8] - F[4]*F[5]*F[6]*F[8] + F[5]*F[5]*F[6]*F[7];
				dF_inv_dF(0,2) = -F[3]*F[4]*F[7]*F[8] + F[3]*F[5]*F[7]*F[7] + F[4]*F[4]*F[6]*F[8] - F[4]*F[5]*F[6]*F[7];
				dF_inv_dF(0,3) = F[1]*F[4]*F[8]*F[8] - F[1]*F[5]*F[7]*F[8] - F[2]*F[4]*F[7]*F[8] + F[2]*F[5]*F[7]*F[7];
				dF_inv_dF(0,4) = -F[1]*F[3]*F[8]*F[8] + F[1]*F[5]*F[6]*F[8] + F[2]*F[3]*F[7]*F[8] - F[2]*F[5]*F[6]*F[7];
				dF_inv_dF(0,5) = F[1]*F[3]*F[7]*F[8] - F[1]*F[4]*F[6]*F[8] - F[2]*F[3]*F[7]*F[7] + F[2]*F[4]*F[6]*F[7];
				dF_inv_dF(0,6) = -F[1]*F[4]*F[5]*F[8] + F[1]*F[5]*F[5]*F[7] + F[2]*F[4]*F[4]*F[8] - F[2]*F[4]*F[5]*F[7];
				dF_inv_dF(0,7) = F[1]*F[3]*F[5]*F[8] - F[1]*F[5]*F[5]*F[6] - F[2]*F[3]*F[4]*F[8] + F[2]*F[4]*F[5]*F[6];
				dF_inv_dF(0,8) = -F[1]*F[3]*F[5]*F[7] + F[1]*F[4]*F[5]*F[6] + F[2]*F[3]*F[4]*F[7] - F[2]*F[4]*F[4]*F[6];
				dF_inv_dF(1,0) = F[1]*F[4]*F[8]*F[8] - F[1]*F[5]*F[7]*F[8] - F[2]*F[4]*F[7]*F[8] + F[2]*F[5]*F[7]*F[7];
				dF_inv_dF(1,1) = -F[0]*F[4]*F[8]*F[8] + F[0]*F[5]*F[7]*F[8] + F[2]*F[4]*F[6]*F[8] - F[2]*F[5]*F[6]*F[7];
				dF_inv_dF(1,2) = F[0]*F[4]*F[7]*F[8] - F[0]*F[5]*F[7]*F[7] - F[1]*F[4]*F[6]*F[8] + F[1]*F[5]*F[6]*F[7];
				dF_inv_dF(1,3) = -F[1]*F[1]*F[8]*F[8] + 2.0*F[1]*F[2]*F[7]*F[8] - F[2]*F[2]*F[7]*F[7];
				dF_inv_dF(1,4) = F[0]*F[1]*F[8]*F[8] - F[0]*F[2]*F[7]*F[8] - F[1]*F[2]*F[6]*F[8] + F[2]*F[2]*F[6]*F[7];
				dF_inv_dF(1,5) = -F[0]*F[1]*F[7]*F[8] + F[0]*F[2]*F[7]*F[7] + F[1]*F[1]*F[6]*F[8] - F[1]*F[2]*F[6]*F[7];
				dF_inv_dF(1,6) = F[1]*F[1]*F[5]*F[8] - F[1]*F[2]*F[4]*F[8] - F[1]*F[2]*F[5]*F[7] + F[2]*F[2]*F[4]*F[7];
				dF_inv_dF(1,7) = -F[0]*F[1]*F[5]*F[8] + F[0]*F[2]*F[4]*F[8] + F[1]*F[2]*F[5]*F[6] - F[2]*F[2]*F[4]*F[6];
				dF_inv_dF(1,8) = F[0]*F[1]*F[5]*F[7] - F[0]*F[2]*F[4]*F[7] - F[1]*F[1]*F[5]*F[6] + F[1]*F[2]*F[4]*F[6];
				dF_inv_dF(2,0) = -F[1]*F[4]*F[5]*F[8] + F[1]*F[5]*F[5]*F[7] + F[2]*F[4]*F[4]*F[8] - F[2]*F[4]*F[5]*F[7];
				dF_inv_dF(2,1) = F[0]*F[4]*F[5]*F[8] - F[0]*F[5]*F[5]*F[7] - F[2]*F[3]*F[4]*F[8] + F[2]*F[3]*F[5]*F[7];
				dF_inv_dF(2,2) = -F[0]*F[4]*F[4]*F[8] + F[0]*F[4]*F[5]*F[7] + F[1]*F[3]*F[4]*F[8] - F[1]*F[3]*F[5]*F[7];
				dF_inv_dF(2,3) = F[1]*F[1]*F[5]*F[8] - F[1]*F[2]*F[4]*F[8] - F[1]*F[2]*F[5]*F[7] + F[2]*F[2]*F[4]*F[7];
				dF_inv_dF(2,4) = -F[0]*F[1]*F[5]*F[8] + F[0]*F[2]*F[5]*F[7] + F[1]*F[2]*F[3]*F[8] - F[2]*F[2]*F[3]*F[7];
				dF_inv_dF(2,5) = F[0]*F[1]*F[4]*F[8] - F[0]*F[2]*F[4]*F[7] - F[1]*F[1]*F[3]*F[8] + F[1]*F[2]*F[3]*F[7];
				dF_inv_dF(2,6) = -F[1]*F[1]*F[5]*F[5] + 2.0*F[1]*F[2]*F[4]*F[5] - F[2]*F[2]*F[4]*F[4];
				dF_inv_dF(2,7) = F[0]*F[1]*F[5]*F[5] - F[0]*F[2]*F[4]*F[5] - F[1]*F[2]*F[3]*F[5] + F[2]*F[2]*F[3]*F[4];
				dF_inv_dF(2,8) = -F[0]*F[1]*F[4]*F[5] + F[0]*F[2]*F[4]*F[4] + F[1]*F[1]*F[3]*F[5] - F[1]*F[2]*F[3]*F[4];
				dF_inv_dF(3,0) = F[3]*F[4]*F[8]*F[8] - F[3]*F[5]*F[7]*F[8] - F[4]*F[5]*F[6]*F[8] + F[5]*F[5]*F[6]*F[7];
				dF_inv_dF(3,1) = -F[3]*F[3]*F[8]*F[8] + 2.0*F[3]*F[5]*F[6]*F[8] - F[5]*F[5]*F[6]*F[6];
				dF_inv_dF(3,2) = F[3]*F[3]*F[7]*F[8] - F[3]*F[4]*F[6]*F[8] - F[3]*F[5]*F[6]*F[7] + F[4]*F[5]*F[6]*F[6];
				dF_inv_dF(3,3) = -F[0]*F[4]*F[8]*F[8] + F[0]*F[5]*F[7]*F[8] + F[2]*F[4]*F[6]*F[8] - F[2]*F[5]*F[6]*F[7];
				dF_inv_dF(3,4) = F[0]*F[3]*F[8]*F[8] - F[0]*F[5]*F[6]*F[8] - F[2]*F[3]*F[6]*F[8] + F[2]*F[5]*F[6]*F[6];
				dF_inv_dF(3,5) = -F[0]*F[3]*F[7]*F[8] + F[0]*F[4]*F[6]*F[8] + F[2]*F[3]*F[6]*F[7] - F[2]*F[4]*F[6]*F[6];
				dF_inv_dF(3,6) = F[0]*F[4]*F[5]*F[8] - F[0]*F[5]*F[5]*F[7] - F[2]*F[3]*F[4]*F[8] + F[2]*F[3]*F[5]*F[7];
				dF_inv_dF(3,7) = -F[0]*F[3]*F[5]*F[8] + F[0]*F[5]*F[5]*F[6] + F[2]*F[3]*F[3]*F[8] - F[2]*F[3]*F[5]*F[6];
				dF_inv_dF(3,8) = F[0]*F[3]*F[5]*F[7] - F[0]*F[4]*F[5]*F[6] - F[2]*F[3]*F[3]*F[7] + F[2]*F[3]*F[4]*F[6];
				dF_inv_dF(4,0) = -F[1]*F[3]*F[8]*F[8] + F[1]*F[5]*F[6]*F[8] + F[2]*F[3]*F[7]*F[8] - F[2]*F[5]*F[6]*F[7];
				dF_inv_dF(4,1) = F[0]*F[3]*F[8]*F[8] - F[0]*F[5]*F[6]*F[8] - F[2]*F[3]*F[6]*F[8] + F[2]*F[5]*F[6]*F[6];
				dF_inv_dF(4,2) = -F[0]*F[3]*F[7]*F[8] + F[0]*F[5]*F[6]*F[7] + F[1]*F[3]*F[6]*F[8] - F[1]*F[5]*F[6]*F[6];
				dF_inv_dF(4,3) = F[0]*F[1]*F[8]*F[8] - F[0]*F[2]*F[7]*F[8] - F[1]*F[2]*F[6]*F[8] + F[2]*F[2]*F[6]*F[7];
				dF_inv_dF(4,4) = -F[0]*F[0]*F[8]*F[8] + 2.0*F[0]*F[2]*F[6]*F[8] - F[2]*F[2]*F[6]*F[6];
				dF_inv_dF(4,5) = F[0]*F[0]*F[7]*F[8] - F[0]*F[1]*F[6]*F[8] - F[0]*F[2]*F[6]*F[7] + F[1]*F[2]*F[6]*F[6];
				dF_inv_dF(4,6) = -F[0]*F[1]*F[5]*F[8] + F[0]*F[2]*F[5]*F[7] + F[1]*F[2]*F[3]*F[8] - F[2]*F[2]*F[3]*F[7];
				dF_inv_dF(4,7) = F[0]*F[0]*F[5]*F[8] - F[0]*F[2]*F[3]*F[8] - F[0]*F[2]*F[5]*F[6] + F[2]*F[2]*F[3]*F[6];
				dF_inv_dF(4,8) = -F[0]*F[0]*F[5]*F[7] + F[0]*F[1]*F[5]*F[6] + F[0]*F[2]*F[3]*F[7] - F[1]*F[2]*F[3]*F[6];
				dF_inv_dF(5,0) = F[1]*F[3]*F[5]*F[8] - F[1]*F[5]*F[5]*F[6] - F[2]*F[3]*F[4]*F[8] + F[2]*F[4]*F[5]*F[6];
				dF_inv_dF(5,1) = -F[0]*F[3]*F[5]*F[8] + F[0]*F[5]*F[5]*F[6] + F[2]*F[3]*F[3]*F[8] - F[2]*F[3]*F[5]*F[6];
				dF_inv_dF(5,2) = F[0]*F[3]*F[4]*F[8] - F[0]*F[4]*F[5]*F[6] - F[1]*F[3]*F[3]*F[8] + F[1]*F[3]*F[5]*F[6];
				dF_inv_dF(5,3) = -F[0]*F[1]*F[5]*F[8] + F[0]*F[2]*F[4]*F[8] + F[1]*F[2]*F[5]*F[6] - F[2]*F[2]*F[4]*F[6];
				dF_inv_dF(5,4) = F[0]*F[0]*F[5]*F[8] - F[0]*F[2]*F[3]*F[8] - F[0]*F[2]*F[5]*F[6] + F[2]*F[2]*F[3]*F[6];
				dF_inv_dF(5,5) = -F[0]*F[0]*F[4]*F[8] + F[0]*F[1]*F[3]*F[8] + F[0]*F[2]*F[4]*F[6] - F[1]*F[2]*F[3]*F[6];
				dF_inv_dF(5,6) = F[0]*F[1]*F[5]*F[5] - F[0]*F[2]*F[4]*F[5] - F[1]*F[2]*F[3]*F[5] + F[2]*F[2]*F[3]*F[4];
				dF_inv_dF(5,7) = -F[0]*F[0]*F[5]*F[5] + 2.0*F[0]*F[2]*F[3]*F[5] - F[2]*F[2]*F[3]*F[3];
				dF_inv_dF(5,8) = F[0]*F[0]*F[4]*F[5] - F[0]*F[1]*F[3]*F[5] - F[0]*F[2]*F[3]*F[4] + F[1]*F[2]*F[3]*F[3];
				dF_inv_dF(6,0) = -F[3]*F[4]*F[7]*F[8] + F[3]*F[5]*F[7]*F[7] + F[4]*F[4]*F[6]*F[8] - F[4]*F[5]*F[6]*F[7];
				dF_inv_dF(6,1) = F[3]*F[3]*F[7]*F[8] - F[3]*F[4]*F[6]*F[8] - F[3]*F[5]*F[6]*F[7] + F[4]*F[5]*F[6]*F[6];
				dF_inv_dF(6,2) = -F[3]*F[3]*F[7]*F[7] + 2.0*F[3]*F[4]*F[6]*F[7] - F[4]*F[4]*F[6]*F[6];
				dF_inv_dF(6,3) = F[0]*F[4]*F[7]*F[8] - F[0]*F[5]*F[7]*F[7] - F[1]*F[4]*F[6]*F[8] + F[1]*F[5]*F[6]*F[7];
				dF_inv_dF(6,4) = -F[0]*F[3]*F[7]*F[8] + F[0]*F[5]*F[6]*F[7] + F[1]*F[3]*F[6]*F[8] - F[1]*F[5]*F[6]*F[6];
				dF_inv_dF(6,5) = F[0]*F[3]*F[7]*F[7] - F[0]*F[4]*F[6]*F[7] - F[1]*F[3]*F[6]*F[7] + F[1]*F[4]*F[6]*F[6];
				dF_inv_dF(6,6) = -F[0]*F[4]*F[4]*F[8] + F[0]*F[4]*F[5]*F[7] + F[1]*F[3]*F[4]*F[8] - F[1]*F[3]*F[5]*F[7];
				dF_inv_dF(6,7) = F[0]*F[3]*F[4]*F[8] - F[0]*F[4]*F[5]*F[6] - F[1]*F[3]*F[3]*F[8] + F[1]*F[3]*F[5]*F[6];
				dF_inv_dF(6,8) = -F[0]*F[3]*F[4]*F[7] + F[0]*F[4]*F[4]*F[6] + F[1]*F[3]*F[3]*F[7] - F[1]*F[3]*F[4]*F[6];
				dF_inv_dF(7,0) = F[1]*F[3]*F[7]*F[8] - F[1]*F[4]*F[6]*F[8] - F[2]*F[3]*F[7]*F[7] + F[2]*F[4]*F[6]*F[7];
				dF_inv_dF(7,1) = -F[0]*F[3]*F[7]*F[8] + F[0]*F[4]*F[6]*F[8] + F[2]*F[3]*F[6]*F[7] - F[2]*F[4]*F[6]*F[6];
				dF_inv_dF(7,2) = F[0]*F[3]*F[7]*F[7] - F[0]*F[4]*F[6]*F[7] - F[1]*F[3]*F[6]*F[7] + F[1]*F[4]*F[6]*F[6];
				dF_inv_dF(7,3) = -F[0]*F[1]*F[7]*F[8] + F[0]*F[2]*F[7]*F[7] + F[1]*F[1]*F[6]*F[8] - F[1]*F[2]*F[6]*F[7];
				dF_inv_dF(7,4) = F[0]*F[0]*F[7]*F[8] - F[0]*F[1]*F[6]*F[8] - F[0]*F[2]*F[6]*F[7] + F[1]*F[2]*F[6]*F[6];
				dF_inv_dF(7,5) = -F[0]*F[0]*F[7]*F[7] + 2.0*F[0]*F[1]*F[6]*F[7] - F[1]*F[1]*F[6]*F[6];
				dF_inv_dF(7,6) = F[0]*F[1]*F[4]*F[8] - F[0]*F[2]*F[4]*F[7] - F[1]*F[1]*F[3]*F[8] + F[1]*F[2]*F[3]*F[7];
				dF_inv_dF(7,7) = -F[0]*F[0]*F[4]*F[8] + F[0]*F[1]*F[3]*F[8] + F[0]*F[2]*F[4]*F[6] - F[1]*F[2]*F[3]*F[6];
				dF_inv_dF(7,8) = F[0]*F[0]*F[4]*F[7] - F[0]*F[1]*F[3]*F[7] - F[0]*F[1]*F[4]*F[6] + F[1]*F[1]*F[3]*F[6];
				dF_inv_dF(8,0) = -F[1]*F[3]*F[5]*F[7] + F[1]*F[4]*F[5]*F[6] + F[2]*F[3]*F[4]*F[7] - F[2]*F[4]*F[4]*F[6];
				dF_inv_dF(8,1) = F[0]*F[3]*F[5]*F[7] - F[0]*F[4]*F[5]*F[6] - F[2]*F[3]*F[3]*F[7] + F[2]*F[3]*F[4]*F[6];
				dF_inv_dF(8,2) = -F[0]*F[3]*F[4]*F[7] + F[0]*F[4]*F[4]*F[6] + F[1]*F[3]*F[3]*F[7] - F[1]*F[3]*F[4]*F[6];
				dF_inv_dF(8,3) = F[0]*F[1]*F[5]*F[7] - F[0]*F[2]*F[4]*F[7] - F[1]*F[1]*F[5]*F[6] + F[1]*F[2]*F[4]*F[6];
				dF_inv_dF(8,4) = -F[0]*F[0]*F[5]*F[7] + F[0]*F[1]*F[5]*F[6] + F[0]*F[2]*F[3]*F[7] - F[1]*F[2]*F[3]*F[6];
				dF_inv_dF(8,5) = F[0]*F[0]*F[4]*F[7] - F[0]*F[1]*F[3]*F[7] - F[0]*F[1]*F[4]*F[6] + F[1]*F[1]*F[3]*F[6];
				dF_inv_dF(8,6) = -F[0]*F[1]*F[4]*F[5] + F[0]*F[2]*F[4]*F[4] + F[1]*F[1]*F[3]*F[5] - F[1]*F[2]*F[3]*F[4];
				dF_inv_dF(8,7) = F[0]*F[0]*F[4]*F[5] - F[0]*F[1]*F[3]*F[5] - F[0]*F[2]*F[3]*F[4] + F[1]*F[2]*F[3]*F[3];
				dF_inv_dF(8,8) = -F[0]*F[0]*F[4]*F[4] + 2.0*F[0]*F[1]*F[3]*F[4] - F[1]*F[1]*F[3]*F[3];
				dF_inv_dF *= 1.0/get_J(F)/get_J(F);

				dd_dF_inv.Tvmult(d_dd_dF_inv, d);
				dd_dF_dot.Tmmult(d2delta_dF_dot_dF_inv, dd_dF_inv);
				d2delta_dF_dot_dF_inv *= dh;
				for(unsigned int i = 0; i < 9; ++i)
					for(unsigned int j = 0; j < 9; ++j)
						d2delta_dF_dot_dF_inv(i,j) += 2.0 * d2h * d_dd_dF_dot[i] * d_dd_dF_inv[j] + dh * d_d2d_dF_dot_d_F_inv(i,j);
				d2delta_dF_dot_dF_inv.mmult(d2delta_dF_dot_dF, dF_inv_dF);
				for(unsigned int i = 0; i < 9; ++i)
					for(unsigned int j = 0; j < 9; ++j)
						d2_omega(i,j + 9) = d2delta_dF_dot_dF(i,j);
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
		static dealii::Vector<double> F(9);
		while(true)
		{

			for(unsigned int m = 0; m < 9; ++m)
				F[m] = e_omega[m] + factor * delta_e_omega[m];
			if( (get_J(F) > 0.0) )
				break;

			factor *= 0.5;
			Assert(factor > 0.0, dealii::ExcMessage("Cannot determine a positive scaling of the load step such that the determinant of the deformation gradient and that of Q stays positive!"));
		}
		return factor;
	}

};

/**
 * Class defining Lagrangian multiplier term coupling fluid velocity to fluid flux.
 *
 * \f$ \omega^\Omega =	\boldsymbol{t} \cdot \left( \boldsymbol{v} - \dot{\boldsymbol{u}} - \dfrac{1}{c} \boldsymbol{F}\cdot \boldsymbol{I} \right)\f$,
 *
 * where \f$t\f$ is the Lagrangian multiplier,<br>
 * \f$\boldsymbol{v}\f$ the fluid velocity,<br>
 * \f$\boldsymbol{u}\f$ the velocity of the moving fictitious skeleton,<br>
 * \f$c\f$ the fluid concentration,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient related to the motion of the fictitious skeleton,<br>
 * and \f$\boldsymbol{I}\f$ is the fluid flux.
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{u}_x\f$<br>
 * 																				[1]	 \f$\dot{u}_y\f$<br>
 * 																				[2]	 \f$\dot{u}_z\f$<br>
 *																				[3]  \f$v_x\f$<br>
 * 																				[4]  \f$v_y\f$<br>
 * 																				[5]  \f$v_z\f$<br>
 * 																				[6]  \f$I_x\f$<br>
 * 																				[7]  \f$I_y\f$<br>
 * 																				[8]  \f$I_z\f$<br>
 * 																				[9]	 \f$t_x\f$<br>
 * 																				[10] \f$t_y\f$<br>
 * 																				[11] \f$t_z\f$<br>
 * 																				[12] \f$c\f$<br>
 * 																				[13] \f$F_{xx}\f$<br>
 * 																				[14] \f$F_{xy}\f$<br>
 * 																				[15] \f$F_{xz}\f$<br>
 * 																				[16] \f$F_{yx}\f$<br>
 * 																				[17] \f$F_{yy}\f$<br>
 * 																				[18] \f$F_{yz}\f$<br>
 * 																				[19] \f$F_{zx}\f$<br>
 * 																				[20] \f$F_{zy}\f$<br>
 * 																				[21] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaVelocityFlux00 : public incrementalFE::Omega<spacedim, spacedim>
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
	OmegaVelocityFlux00(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 3, 6, 3, 10, method, alpha, "OmegaVelocityFlux00")
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
		const unsigned int i_u_dot = 0;
		const unsigned int i_v = 3;
		const unsigned int i_I = 6;
		const unsigned int i_t = 9;
		const unsigned int i_c = 12;
		const unsigned int i_F = 13;

		dealii::Tensor<1,3> u_dot, v, I, t_;
		dealii::Tensor<2,3> F;
		for(unsigned int i = 0; i < 3; ++i)
		{
			u_dot[i] = values[i_u_dot + i];
			v[i] = values[i_v + i];
			I[i] = values[i_I + i];
			t_[i] = values[i_t + i];
		}
		const double c = values[i_c];
		for(unsigned int i = 0; i < 3; ++i)
			for(unsigned int j = 0; j < 3; ++j)
				F[i][j] = values[i_F + i * 3 + j];

		dealii::Tensor<1,3> c_F_I = 1.0/c * F * I;
		dealii::Tensor<1,3> c_t_F = 1.0/c * transpose(F) * t_;

		if(get<0>(requested_quantities))
		{
			omega = t_ * (v - u_dot - c_F_I);
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int i = 0; i < 3; ++i)
			{
				d_omega[i_u_dot + i] = -t_[i];
				d_omega[i_v + i] = t_[i];
				d_omega[i_I + i] = -c_t_F[i];
				d_omega[i_t + i] = (v - u_dot - c_F_I)[i];
			}
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int i = 0; i < 3; ++i)
			{
				d2_omega[i_u_dot + i][i_t + i] = d2_omega[i_t + i][i_u_dot + i] = -1.0;
				d2_omega[i_v + i][i_t + i] = d2_omega[i_t + i][i_v + i] = 1.0;
				for(unsigned int j = 0; j < 3; ++j)
					d2_omega[i_I + i][i_t + j] = d2_omega[i_t + j][i_I + i] = -1.0 / c * F[j][i];
			}
		}

		return false;

	}
};

/**
 * Class defining Lagrangian multiplier term coupling fluid velocity to fluid flux.
 *
 * \f$ \omega^\Omega =	\boldsymbol{t} \cdot \left( \boldsymbol{v} - \dot{\boldsymbol{u}} - \dfrac{V^\mathrm{f}_\mathrm{m}}{J} \boldsymbol{F}\cdot \boldsymbol{I} \right)\f$,
 *
 * where \f$t\f$ is the Lagrangian multiplier,<br>
 * \f$\boldsymbol{v}\f$ the fluid velocity,<br>
 * \f$\boldsymbol{u}\f$ the velocity of the moving fictitious skeleton,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient related to the motion of the fictitious skeleton,<br>
 * and \f$\boldsymbol{I}\f$ is the fluid flux.
 *
 * @warning Currently, the derivatives required for the \f$\alpha\f$-family for temporal discretization are not implemented!
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{u}_x\f$<br>
 * 																				[1]	 \f$\dot{u}_y\f$<br>
 * 																				[2]	 \f$\dot{u}_z\f$<br>
 *																				[3]  \f$v_x\f$<br>
 * 																				[4]  \f$v_y\f$<br>
 * 																				[5]  \f$v_z\f$<br>
 * 																				[6]  \f$I_x\f$<br>
 * 																				[7]  \f$I_y\f$<br>
 * 																				[8]  \f$I_z\f$<br>
 * 																				[9]	 \f$t_x\f$<br>
 * 																				[10] \f$t_y\f$<br>
 * 																				[11] \f$t_z\f$<br>
 * 																				[12] \f$F_{xx}\f$<br>
 * 																				[13] \f$F_{xy}\f$<br>
 * 																				[14] \f$F_{xz}\f$<br>
 * 																				[15] \f$F_{yx}\f$<br>
 * 																				[16] \f$F_{yy}\f$<br>
 * 																				[17] \f$F_{yz}\f$<br>
 * 																				[18] \f$F_{zx}\f$<br>
 * 																				[19] \f$F_{zy}\f$<br>
 * 																				[20] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaVelocityFlux01 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * molar volume of fluid
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
	 * @param[in]		V_m_f					OmegaVelocityFlux01::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaVelocityFlux01(const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																V_m_f,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 3, 6, 3, 9, method, alpha, "OmegaVelocityFlux01"),
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

		// start indices for respective quantities
		const unsigned int i_u_dot = 0;
		const unsigned int i_v = 3;
		const unsigned int i_I = 6;
		const unsigned int i_t = 9;
		const unsigned int i_F = 12;

		dealii::Tensor<1,3> u_dot, v, I, t_;
		dealii::Tensor<2,3> F;
		for(unsigned int i = 0; i < 3; ++i)
		{
			u_dot[i] = values[i_u_dot + i];
			v[i] = values[i_v + i];
			I[i] = values[i_I + i];
			t_[i] = values[i_t + i];
		}
		for(unsigned int i = 0; i < 3; ++i)
			for(unsigned int j = 0; j < 3; ++j)
				F[i][j] = values[i_F + i * 3 + j];

		const double J = determinant(F);
		const double c = J / V_m_f;

		dealii::Tensor<1,3> c_F_I = 1.0/c * F * I;
		dealii::Tensor<1,3> c_t_F = 1.0/c * transpose(F) * t_;

		if(get<0>(requested_quantities))
		{
			omega = t_ * (v - u_dot - c_F_I);
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int i = 0; i < 3; ++i)
			{
				d_omega[i_u_dot + i] = -t_[i];
				d_omega[i_v + i] = t_[i];
				d_omega[i_I + i] = -c_t_F[i];
				d_omega[i_t + i] = (v - u_dot - c_F_I)[i];
			}
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int i = 0; i < 3; ++i)
			{
				d2_omega[i_u_dot + i][i_t + i] = d2_omega[i_t + i][i_u_dot + i] = -1.0;
				d2_omega[i_v + i][i_t + i] = d2_omega[i_t + i][i_v + i] = 1.0;
				for(unsigned int j = 0; j < 3; ++j)
					d2_omega[i_I + i][i_t + j] = d2_omega[i_t + j][i_I + i] = -1.0 / c * F[j][i];
			}
			if(compute_dq)
			{
				dealii::Tensor<2,3> F_inv = invert(F);
				for(unsigned int k = 0; k < 3; ++k)
				{
					for(unsigned int r = 0; r < 3; ++r)
					{
						for(unsigned int R = 0; R < 3; ++R)
						{
							d2_omega(i_I + k, i_F + 3*r + R) = c_t_F[k] * F_inv[R][r];
							d2_omega(i_t + k, i_F + 3*r + R) = c_F_I[k] * F_inv[R][r];
							if(R == k)
								d2_omega(i_I + k, i_F + 3*r + R) += -1.0/c * t_[r];
							if(r == k)
								d2_omega(i_t + k, i_F + 3*r + R) += -1.0/c * I[R];

						}
					}
				}

			}
		}

		return false;

	}
};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	\dfrac{J \eta}{2} (\boldsymbol{d}:\boldsymbol{d}) \f$,
 *
 * where \f$\boldsymbol{d} = \dfrac{1}{2}\left( \boldsymbol{F}^{-1} \cdot \nabla \boldsymbol{v} + \nabla \boldsymbol{v} \cdot \boldsymbol{F}^{-1} \right)\f$ is the stretching,
 * \f$\boldsymbol{F}\f$ is the deformation gradient associated with the background solid skeleton, \f$eta\f$ is the viscosity, and \f$J = \det\boldsymbol{F}\f$.
 *
 * This describes a viscous fluid.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$v_{x,x}\f$<br>
 * 																				[1]  \f$v_{x,y}\f$<br>
 * 																				[2]  \f$v_{x,z}\f$<br>
 * 																				[3]  \f$v_{y,x}\f$<br>
 * 																				[4]  \f$v_{y,y}\f$<br>
 * 																				[5]  \f$v_{y,z}\f$<br>
 * 																				[6]  \f$v_{z,x}\f$<br>
 * 																				[7]  \f$v_{z,y}\f$<br>
 * 																				[8]  \f$v_{z,z}\f$<br>
 * 																				[9]  \f$F_{xx}\f$<br>
 * 																				[10] \f$F_{xy}\f$<br>
 * 																				[11] \f$F_{xz}\f$<br>
 * 																				[12] \f$F_{yx}\f$<br>
 * 																				[13] \f$F_{yy}\f$<br>
 * 																				[14] \f$F_{yz}\f$<br>
 * 																				[15] \f$F_{zx}\f$<br>
 * 																				[16] \f$F_{zy}\f$<br>
 * 																				[17] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaViscousDissipation02 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * material parameter
	 */
	const double
	eta;

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
	 * @param[in]		eta						OmegaViscousDissipation01::eta
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaViscousDissipation02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>									domain_of_integration,
								const dealii::Quadrature<spacedim>											quadrature,
								GlobalDataIncrementalFE<spacedim>&											global_data,
								const double																eta,
								const unsigned int															method,
								const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 0, 9, 0, 9, method, alpha, "OmegaViscousDissipation02"),
	eta(eta)
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

		dealii::Tensor<2,3> grad_ref_v, F, F_inv, C_inv, grad_v, delta;
		for(unsigned int r = 0; r < 3; ++r)
		{
			for(unsigned int S = 0; S < 3; ++S)
			{
				grad_ref_v[r][S] = values[3 * r + S];
				F[r][S] = values[9 + 3 * r + S];
			}
		}
		F_inv = invert(F);
		C_inv = F_inv * transpose(F_inv);
		grad_v = grad_ref_v * F_inv;
		delta[0][0] = delta[1][1] = delta[2][2] = 1.0;
		const double J = determinant(F);

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * J * eta * ( trace(grad_v * transpose(grad_v)) + trace(grad_v * grad_v) );
		}

		dealii::Tensor<2,3> T;
		if(get<1>(requested_quantities) || get<2>(requested_quantities))
		{
			T = J * eta * (grad_v + transpose(grad_v)) * transpose(F_inv);
			if(get<1>(requested_quantities))
			{
				for(unsigned int r = 0; r < 3; ++r)
					for(unsigned int S = 0; S < 3; ++S)
						d_omega[r * 3 + S] = T[r][S];
			}
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int S = 0; S < 3; ++S)
					for(unsigned int p = 0; p < 3; ++p)
						for(unsigned int Q = 0; Q < 3; ++Q)
							d2_omega(r * 3 + S, p * 3 + Q) = J * eta * (delta[r][p] * C_inv[S][Q] + F_inv[Q][r] * F_inv[S][p]);
			if(compute_dq)
			{
				dealii::Tensor<2,3> F_inv_grad_v, F_inv_grad_v_T;
				F_inv_grad_v = F_inv * grad_v;
				F_inv_grad_v_T = F_inv * transpose(grad_v);

				for(unsigned int r = 0; r < 3; ++r)
					for(unsigned int S = 0; S < 3; ++S)
						for(unsigned int p = 0; p < 3; ++p)
							for(unsigned int Q = 0; Q < 3; ++Q)
								d2_omega(r * 3 + S, p * 3 + Q + 9) = F_inv[Q][p] * T[r][S] - J * eta * ( F_inv_grad_v[S][p] * F_inv[Q][r] + C_inv[Q][S] * grad_v[r][p] + ( F_inv_grad_v[Q][r] + F_inv_grad_v_T[Q][r] ) * F_inv[S][p] );
			}
		}

		return false;
	}
};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	-J \boldsymbol{F}^{-\top} : \nabla \boldsymbol{f} \dot{\xi} \f$,
 *
 * where \f$\boldsymbol{F}\f$ is the deformation gradient, \f$J\f$ its determinant,  \f$boldsymbol{f}\f$ a Lagrangian multiplier vector
 * and \f$\dot{\xi}\f$ the rate of a scalar field.
 *
 * This defines an "incompressibility" condition for the Lagrangian multiplier.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{\xi}\f$<br>
 * 																				[1]  \f$f_{x,x}\f$<br>
 * 																				[2]  \f$f_{x,y}\f$<br>
 * 																				[3]  \f$f_{x,z}\f$<br>
 * 																				[4]  \f$f_{y,x}\f$<br>
 * 																				[5]  \f$f_{y,y}\f$<br>
 * 																				[6]  \f$f_{y,z}\f$<br>
 * 																				[7]  \f$f_{z,x}\f$<br>
 * 																				[8]  \f$f_{z,y}\f$<br>
 * 																				[9]  \f$f_{z,z}\f$<br>
 * 																				[10] \f$F_{xx}\f$<br>
 * 																				[11] \f$F_{xy}\f$<br>
 * 																				[12] \f$F_{xz}\f$<br>
 * 																				[13] \f$F_{yx}\f$<br>
 * 																				[14] \f$F_{yy}\f$<br>
 * 																				[15] \f$F_{yz}\f$<br>
 * 																				[16] \f$F_{zx}\f$<br>
 * 																				[17] \f$F_{zy}\f$<br>
 * 																				[18] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaLagrangeIncompressibility00 : public incrementalFE::Omega<spacedim, spacedim>
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
	OmegaLagrangeIncompressibility00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
										const std::set<dealii::types::material_id>									domain_of_integration,
										const dealii::Quadrature<spacedim>											quadrature,
										GlobalDataIncrementalFE<spacedim>&											global_data,
										const unsigned int															method,
										const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 0, 9, 9, method, alpha, "OmegaLagrangeIncompressibility00")
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

		dealii::Tensor<2,3> grad_ref_f, F, F_inv, grad_f;

		const double xi_dot = values[0];
		for(unsigned int r = 0; r < 3; ++r)
		{
			for(unsigned int S = 0; S < 3; ++S)
			{
				grad_ref_f[r][S] = values[1 + 3 * r + S];
				F[r][S] = values[1 + 9 + 3 * r + S];
			}
		}
		F_inv = invert(F);
		grad_f = grad_ref_f * F_inv;
		const double J = determinant(F);
		const double tr_grad_f = trace(grad_f);

		if(get<0>(requested_quantities))
		{
			omega = -J * tr_grad_f * xi_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[0] = -J *tr_grad_f;
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int R = 0; R < 3; ++R)
					d_omega[1 + r * 3 + R] = -J * F_inv[R][r] * xi_dot;
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int R = 0; R < 3; ++R)
					d2_omega(0, 1 + r * 3 + R) = d2_omega(1 + r * 3 + R, 0) = -J * F_inv[R][r];

			if(compute_dq)
			{
				dealii::Tensor<2,3> F_inv_grad_f = F_inv * grad_f;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_omega(0, 1 + 9 + r * 3 + R) = -J * ( tr_grad_f * F_inv[R][r] - F_inv_grad_f[R][r]);
						for(unsigned int k = 0; k < 3; ++k)
							for(unsigned int K = 0; K < 3; ++K)
								d2_omega(1 + k * 3 + K, 1 + 9 + r * 3 + R) = -J * xi_dot * ( F_inv[R][r] * F_inv[K][k] - F_inv[K][r] * F_inv[R][k] );
					}
				}
			}
		}

		return false;
	}
};


/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	J \boldsymbol{f} \cdot \boldsymbol{F}^{-\top} \cdot \nabla \dot{\xi} \f$,
 *
 * where \f$\boldsymbol{F}\f$ is the deformation gradient, \f$J\f$ its determinant,  \f$boldsymbol{f}\f$ a Lagrangian multiplier vector
 * and \f$\dot{\xi}\f$ the rate of a scalar field.
 *
 * This defines an "incompressibility" condition for the Lagrangian multiplier.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{\xi}_x\f$<br>
 * 																				[1]  \f$\dot{\xi}_y\f$<br>
 * 																				[2]  \f$\dot{\xi}_z\f$<br>
 * 																				[3]  \f$f_x\f$<br>
 * 																				[4]  \f$f_y\f$<br>
 * 																				[5]  \f$f_z\f$<br>
 * 																				[6] \f$F_{xx}\f$<br>
 * 																				[7] \f$F_{xy}\f$<br>
 * 																				[8] \f$F_{xz}\f$<br>
 * 																				[9] \f$F_{yx}\f$<br>
 * 																				[10] \f$F_{yy}\f$<br>
 * 																				[11] \f$F_{yz}\f$<br>
 * 																				[12] \f$F_{zx}\f$<br>
 * 																				[13] \f$F_{zy}\f$<br>
 * 																				[14] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaLagrangeIncompressibility01 : public incrementalFE::Omega<spacedim, spacedim>
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
	OmegaLagrangeIncompressibility01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
										const std::set<dealii::types::material_id>									domain_of_integration,
										const dealii::Quadrature<spacedim>											quadrature,
										GlobalDataIncrementalFE<spacedim>&											global_data,
										const unsigned int															method,
										const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 3, 0, 3, 9, method, alpha, "OmegaLagrangeIncompressibility01")
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

		dealii::Tensor<2,3> F, F_inv;
		dealii::Tensor<1,3> grad_xi_dot_ref, f_def, f_ref, grad_xi_dot_def;

		for(unsigned int r = 0; r < 3; ++r)
		{
			grad_xi_dot_ref[r] = values[r];
			f_def[r] = values[3 + r];
			for(unsigned int S = 0; S < 3; ++S)
				F[r][S] = values[6 + 3 * r + S];
		}
		F_inv = invert(F);
		const double J = determinant(F);

		f_ref = J * F_inv * f_def;
		grad_xi_dot_def = J * transpose(F_inv) * grad_xi_dot_ref;

		if(get<0>(requested_quantities))
		{
			omega = f_ref * grad_xi_dot_ref;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
			{
				d_omega[r] = f_ref[r];
				d_omega[r + 3] = grad_xi_dot_def[r];
			}
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int R = 0; R < 3; ++R)
				for(unsigned int r = 0; r < 3; ++r)
					d2_omega(R, 3 + r) = d2_omega(3 + r, R) = J * F_inv[R][r];

			if(compute_dq)
			{
				for(unsigned int k = 0; k < 3; ++k)
				{
					for(unsigned int r = 0; r < 3; ++r)
					{
						for(unsigned int R = 0; R < 3; ++R)
						{
							d2_omega(k, 6 + 3 * r + R) = F_inv[R][r] * f_ref[k] - F_inv[k][r] * f_ref[R];
							d2_omega(k + 3, 6 + 3 * r + R) = F_inv[R][r] * grad_xi_dot_def[k] - F_inv[R][k] * grad_xi_dot_def[r];
						}
					}
				}
			}
		}

		return false;
	}
};


/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	J \boldsymbol{f} \cdot \boldsymbol{F}^{-\top} \cdot \boldsymbol{N} \dot{\xi} \f$,
 *
 * where \f$\boldsymbol{F}\f$ is the deformation gradient, \f$J\f$ its determinant,  \f$boldsymbol{f}\f$ a Lagrangian multiplier vector,
 * \f$\dot{\xi}\f$ the rate of a scalar field, and \f$\boldsymbol{N}\f$ the unit normal vector.
 *
 * This is the interfacial counterpart to OmegaLagrangeIncompressibility00.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma: <br>[0]  \f$\xi\f$<br>
 * 															[1]  \f$f_x\f$<br>
 * 															[2]  \f$f_y\f$<br>
 * 															[3]  \f$f_z\f$<br>
 * 															[4]  \f$F_{xx}\f$<br>
 * 															[5]  \f$F_{xy}\f$<br>
 * 															[6]  \f$F_{xz}\f$<br>
 * 															[7]  \f$F_{yx}\f$<br>
 * 															[8] \f$F_{yy}\f$<br>
 * 															[9] \f$F_{yz}\f$<br>
 * 															[10] \f$F_{zx}\f$<br>
 * 															[11] \f$F_{zy}\f$<br>
 * 															[12] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaLagrangeIncompressibility02 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * orientation of normal vector (if false, the normal vector points from - to +, if true the other way round)
	 */
	const bool
	flip_normal;

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
	 * @param[in]		flip_normal				OmegaLagrangeIncompressibility02::flip_normal
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaLagrangeIncompressibility02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
										const std::set<dealii::types::material_id>										domain_of_integration,
										const dealii::Quadrature<spacedim-1>											quadrature,
										GlobalDataIncrementalFE<spacedim>&												global_data,
										const bool																		flip_normal,
										const unsigned int																method,
										const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 0, 3, 9, method, alpha, "OmegaLagrangeIncompressibility02"),
	flip_normal(flip_normal)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{

		dealii::Tensor<2,3> F, F_inv;

		const double xi_dot = values[0];
		dealii::Tensor<1,3> f;
		for(unsigned int r = 0; r < 3; ++r)
		{
			f[r] = values[1 + r];
			for(unsigned int S = 0; S < 3; ++S)
				F[r][S] = values[4 + 3 * r + S];
		}
		F_inv = invert(F);
		const double J = determinant(F);

		dealii::Tensor<1, 3> n__, n_;
		n__[0] = n[0];
		n__[1] = spacedim < 2 ? 0.0 : n[1];
		n__[2] = spacedim < 3 ? 0.0 : n[2];
		n_ = flip_normal ? -J * transpose(F_inv) * n__ : J * transpose(F_inv) * n__;

		if(get<0>(requested_quantities))
		{
			sigma = xi_dot * f * n_;
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = f * n_;
			for(unsigned int r = 0; r < 3; ++r)
				d_sigma[1 + r] =  xi_dot * n_[r];
		}

		if(get<2>(requested_quantities))
		{

			for(unsigned int r = 0; r < 3; ++r)
				d2_sigma(0, 1 + r) = d2_sigma(1 + r, 0) = n_[r];

			if(compute_dq)
			{
				dealii::Tensor<3,3> dn_dF;
				dealii::Tensor<2,3> f_dn_dF;
				for(unsigned int m = 0; m < 3; ++m)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							dn_dF[m][r][R] = F_inv[R][r] * n_[m] - F_inv[R][m] * n_[r];
				f_dn_dF = f * dn_dF;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_sigma(0, 4 + r * 3 + R) = f_dn_dF[r][R];
						for(unsigned int k = 0; k < 3; ++k)
							d2_sigma(1 + k, 4 + r * 3 + R) = xi_dot * dn_dF[k][r][R];
					}
				}
			}
		}

		return false;
	}
};


/**
 * Class defining an interface-related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	-J \boldsymbol{F}^{-\top} : \nabla \boldsymbol{f} \dot{\xi} \f$,
 *
 * where \f$\boldsymbol{F}\f$ is the deformation gradient, \f$J\f$ its determinant,  \f$boldsymbol{f}\f$ a Lagrangian multiplier vector
 * and \f$\dot{\xi}\f$ the rate of a scalar field.
 *
 * This defines an "incompressibility" condition for the Lagrangian multiplier.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$\dot{\xi}\f$<br>
 * 																				[1]  \f$f_{x,x}\f$<br>
 * 																				[2]  \f$f_{x,y}\f$<br>
 * 																				[3]  \f$f_{x,z}\f$<br>
 * 																				[4]  \f$f_{y,x}\f$<br>
 * 																				[5]  \f$f_{y,y}\f$<br>
 * 																				[6]  \f$f_{y,z}\f$<br>
 * 																				[7]  \f$f_{z,x}\f$<br>
 * 																				[8]  \f$f_{z,y}\f$<br>
 * 																				[9]  \f$f_{z,z}\f$<br>
 * 																				[10] \f$F_{xx}\f$<br>
 * 																				[11] \f$F_{xy}\f$<br>
 * 																				[12] \f$F_{xz}\f$<br>
 * 																				[13] \f$F_{yx}\f$<br>
 * 																				[14] \f$F_{yy}\f$<br>
 * 																				[15] \f$F_{yz}\f$<br>
 * 																				[16] \f$F_{zx}\f$<br>
 * 																				[17] \f$F_{zy}\f$<br>
 * 																				[18] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaLagrangeIncompressibility03 : public incrementalFE::Omega<spacedim-1, spacedim>
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
	OmegaLagrangeIncompressibility03(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
										const std::set<dealii::types::material_id>										domain_of_integration,
										const dealii::Quadrature<spacedim-1>											quadrature,
										GlobalDataIncrementalFE<spacedim>&												global_data,
										const unsigned int																method,
										const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 1, 0, 9, 9, method, alpha, "OmegaLagrangeIncompressibility03")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{

		dealii::Tensor<2,3> grad_ref_f, F, F_inv, grad_f;

		const double xi_dot = values[0];
		for(unsigned int r = 0; r < 3; ++r)
		{
			for(unsigned int S = 0; S < 3; ++S)
			{
				grad_ref_f[r][S] = values[1 + 3 * r + S];
				F[r][S] = values[1 + 9 + 3 * r + S];
			}
		}
		F_inv = invert(F);
		grad_f = grad_ref_f * F_inv;
		const double J = determinant(F);
		const double tr_grad_f = trace(grad_f);

		if(get<0>(requested_quantities))
		{
			sigma = -J * tr_grad_f * xi_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = -J *tr_grad_f;
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int R = 0; R < 3; ++R)
					d_sigma[1 + r * 3 + R] = -J * F_inv[R][r] * xi_dot;
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int R = 0; R < 3; ++R)
					d2_sigma(0, 1 + r * 3 + R) = d2_sigma(1 + r * 3 + R, 0) = -J * F_inv[R][r];

			if(compute_dq)
			{
				dealii::Tensor<2,3> F_inv_grad_f = F_inv * grad_f;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_sigma(0, 1 + 9 + r * 3 + R) = -J * ( tr_grad_f * F_inv[R][r] - F_inv_grad_f[R][r]);
						for(unsigned int k = 0; k < 3; ++k)
							for(unsigned int K = 0; K < 3; ++K)
								d2_sigma(1 + k * 3 + K, 1 + 9 + r * 3 + R) = -J * xi_dot * ( F_inv[R][r] * F_inv[K][k] - F_inv[K][r] * F_inv[R][k] );
					}
				}
			}
		}

		return false;
	}
};



/**
 * Class defining Lagrangian multiplier term for an incompressible fluid.
 *
 * \f$ \omega^\Omega =	\left[ \dfrac{J}{V^\mathrm{f}_\mathrm{m}}\boldsymbol{F}^{-1} \cdot \left( \dot{\boldsymbol{v}} - \dot{\boldsymbol{u}} \right) \right] \cdot  \nabla \eta^\mathrm{f} - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * and \f$\eta^\mathrm{f}\f$ is the fluid potential<br>
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  			\f$\dot{v}_{x}\f$<br>
 * 																				[1]				\f$\dot{v}_{y}\f$<br>
 * 																				[2]				\f$\dot{v}_{z}\f$<br>
 *																				[3]  			\f$\dot{c}^\mathrm{f}\f$<br>
 *																				[4]				\f$\dot{u}_x\f$<br>
 *																				[5]				\f$\dot{u}_y\f$<br>
 *																				[6]				\f$\dot{u}_z\f$<br>
 * 																				[7]  			\f$\eta^\mathrm{f}_{,x}\f$<br>
 * 																				[8]  			\f$\eta^\mathrm{f}_{,y}\f$<br>
 * 																				[9]  			\f$\eta^\mathrm{f}_{,z}\f$<br>
 * 																				[10]			\f$\eta^\mathrm{f}\f$<br>
 * 																				[11] ... [19]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidIncompressibility00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * Number of ionic species \f$V^\mathrm{f}_\mathrm{m}\f$
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
	 * @param[in]		V_m_f					OmegaFluidIncompressibility00::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluidIncompressibility00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>										domain_of_integration,
									const dealii::Quadrature<spacedim>												quadrature,
									GlobalDataIncrementalFE<spacedim>&												global_data,
									const double																	V_m_f,
									const unsigned int																method,
									const double																	alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 4, 3, 4, 9, method, alpha, "OmegaDualFluidIncompressibility00"),
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

		// start indices for respective quantities
		const unsigned int i_v_dot = 0;
		const unsigned int i_c_f_dot = 3;
		const unsigned int i_u_dot = 4;
		const unsigned int i_grad_eta_f = 7;
		const unsigned int i_eta_f = 10;
		const unsigned int i_F = 11;

		dealii::Tensor<1,3> v_dot;
		for(unsigned int m = 0; m < 3; ++m)
			v_dot[m] = values[i_v_dot + m];

		const double c_f_dot = values[i_c_f_dot];

		dealii::Tensor<1,3> u_dot;
		for(unsigned int m = 0; m < 3; ++m)
			u_dot[m] = values[i_u_dot + m];

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		const double J = determinant(F);
		const double c_f = J / V_m_f;

		dealii::Tensor<1,3> grad_eta_c_f, grad_eta_f;
		for(unsigned int m = 0; m < 3; ++m)
		{
			grad_eta_c_f[m] = c_f * values[i_grad_eta_f + m];
			grad_eta_f[m] = values[i_grad_eta_f + m];
		}

		dealii::Tensor<2,3> F_inv;
		dealii::Tensor<1,3> F_inv_delta_v_dot, grad_eta_f_F_inv, grad_eta_c_f_F_inv, delta_v_dot;
		F_inv = invert(F);
		delta_v_dot = v_dot - u_dot;
		F_inv_delta_v_dot = F_inv * delta_v_dot;
		grad_eta_f_F_inv = transpose(F_inv) * grad_eta_f;
		grad_eta_c_f_F_inv = transpose(F_inv) * grad_eta_c_f;

		if(get<0>(requested_quantities))
		{
			omega = F_inv_delta_v_dot * grad_eta_c_f - eta_f * c_f_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_c_f_dot] = -eta_f;
			d_omega[i_eta_f]   = -c_f_dot;
			for(unsigned int m = 0; m < 3; ++m)
			{
				d_omega[i_v_dot + m] = grad_eta_c_f_F_inv[m];
				d_omega[i_u_dot + m] = -grad_eta_c_f_F_inv[m];
				d_omega[i_grad_eta_f + m] = F_inv_delta_v_dot[m] * c_f;
			}
		}

		if(get<2>(requested_quantities))
		{

			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int m = 0; m < 3; ++m)
			{
				for(unsigned int n = 0; n < 3; ++n)
				{
					d2_omega(i_v_dot + m, i_grad_eta_f + n) = d2_omega(i_grad_eta_f + n, i_v_dot + m) = F_inv[n][m] * c_f;
					d2_omega(i_u_dot + m, i_grad_eta_f + n) = d2_omega(i_grad_eta_f + n, i_u_dot + m) = -F_inv[n][m] * c_f;
				}
			}


			if(compute_dq)
			{
				for(unsigned int m = 0; m < 3; ++m)
				{
					for(unsigned int k = 0; k < 3; ++k)
					{
						for(unsigned int l = 0; l < 3; ++l)
						{
							d2_omega(i_v_dot + m, i_F + 3 * k + l) = -grad_eta_c_f_F_inv[k] * F_inv[l][m] + grad_eta_c_f_F_inv[m] * F_inv[l][k];
							d2_omega(i_u_dot + m, i_F + 3 * k + l) = grad_eta_c_f_F_inv[k] * F_inv[l][m] -  grad_eta_c_f_F_inv[m] * F_inv[l][k];
							d2_omega(i_grad_eta_f + m, i_F + 3 * k + l) = -F_inv_delta_v_dot[l] * F_inv[m][k] * c_f + F_inv_delta_v_dot[m] * c_f * F_inv[l][k];
						}
					}

				}

			}

		}
		return false;

	}
};


/**
 * Class defining Lagrangian multiplier term for an incompressible fluid.
 *
 * \f$ \omega^\Omega =	-\dfrac{J}{V^\mathrm{f}_\mathrm{m}}\boldsymbol{F}^{-1} : \nabla\left( \dot{\boldsymbol{v}} - \dot{\boldsymbol{u}} \right) \eta^\mathrm{f} - \eta^\mathrm{f} \dot{c}^\mathrm{f}\f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * and \f$\eta^\mathrm{f}\f$ is the fluid potential<br>
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  			\f$\dot{v}_{,xx}\f$<br>
 * 																				[1]				\f$\dot{v}_{,xy}\f$<br>
 * 																				[2]				\f$\dot{v}_{,xz}\f$<br>
 * 																				[3]  			\f$\dot{v}_{,yx}\f$<br>
 * 																				[4]				\f$\dot{v}_{,yy}\f$<br>
 * 																				[5]				\f$\dot{v}_{,yz}\f$<br>
 * 																				[6]  			\f$\dot{v}_{,zx}\f$<br>
 * 																				[7]				\f$\dot{v}_{,zy}\f$<br>
 * 																				[8]				\f$\dot{v}_{,zz}\f$<br>
 *																				[9]  			\f$\dot{c}^\mathrm{f}\f$<br>
 *																				[10]			\f$\dot{u}_{,xx}\f$<br>
 *																				[11]			\f$\dot{u}_{,xy}\f$<br>
 *																				[12]			\f$\dot{u}_{,xz}\f$<br>
 *																				[13]			\f$\dot{u}_{,yx}\f$<br>
 *																				[14]			\f$\dot{u}_{,yy}\f$<br>
 *																				[15]			\f$\dot{u}_{,yz}\f$<br>
 *																				[16]			\f$\dot{u}_{,zx}\f$<br>
 *																				[17]			\f$\dot{u}_{,zy}\f$<br>
 *																				[18]			\f$\dot{u}_{,zz}\f$<br>
 * 																				[19]			\f$\eta^\mathrm{f}\f$<br>
 * 																				[20] ... [28]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidIncompressibility01 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * Number of ionic species \f$V^\mathrm{f}_\mathrm{m}\f$
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
	 * @param[in]		V_m_f					OmegaFluidIncompressibility01::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluidIncompressibility01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>									domain_of_integration,
									const dealii::Quadrature<spacedim>											quadrature,
									GlobalDataIncrementalFE<spacedim>&											global_data,
									const double																V_m_f,
									const unsigned int															method,
									const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 10, 9, 1, 9, method, alpha, "OmegaDualFluidIncompressibility01"),
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

		// start indices for respective quantities
		const unsigned int i_grad_v_dot = 0;
		const unsigned int i_c_f_dot = 9;
		const unsigned int i_grad_u_dot = 10;
		const unsigned int i_eta_f = 19;
		const unsigned int i_F = 20;

		dealii::Tensor<2,3> grad_v_dot;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_v_dot[r][R] = values[i_grad_v_dot + 3 * r + R];

		const double c_f_dot = values[i_c_f_dot];

		dealii::Tensor<2,3> grad_u_dot;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_u_dot[r][R] = values[i_grad_u_dot + 3 * r + R];

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		const double J = determinant(F);
		dealii::Tensor<2,3> F_inv;
		F_inv = invert(F);

		dealii::Tensor<2,3> grad_v_dot_u_dot = grad_v_dot - grad_u_dot;

		if(get<0>(requested_quantities))
		{
			omega = -J/V_m_f * trace(F_inv * (grad_v_dot_u_dot)) * eta_f - eta_f * c_f_dot;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_c_f_dot] = -eta_f;
			d_omega[i_eta_f]   = -c_f_dot - J/V_m_f * trace(F_inv * (grad_v_dot_u_dot));
			for(unsigned int r = 0; r < 3; ++r)
			{
				for(unsigned int R = 0; R < 3; ++R)
				{
					d_omega[i_grad_v_dot + 3 * r + R] = -J / V_m_f * F_inv[R][r] * eta_f;
					d_omega[i_grad_u_dot + 3 * r + R] = J / V_m_f * F_inv[R][r] * eta_f;
				}
			}
		}

		if(get<2>(requested_quantities))
		{

			d2_omega[i_c_f_dot][i_eta_f] = d2_omega[i_eta_f][i_c_f_dot] = -1.0;

			for(unsigned int r = 0; r < 3; ++r)
			{
				for(unsigned int R = 0; R < 3; ++R)
				{
					d2_omega(i_eta_f, i_grad_v_dot + 3 * r + R) = d2_omega(i_grad_v_dot + 3 * r + R, i_eta_f) = -J / V_m_f * F_inv[R][r];
					d2_omega(i_eta_f, i_grad_u_dot + 3 * r + R) = d2_omega(i_grad_u_dot + 3 * r + R, i_eta_f) = J / V_m_f * F_inv[R][r];
				}
			}


			if(compute_dq)
			{
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						for(unsigned int s = 0; s < 3; ++s)
						{
							for(unsigned int S = 0; S < 3; ++S)
							{
								 d2_omega(i_grad_v_dot + 3 * r + R, i_F + 3 * s + S) = -J/V_m_f * (F_inv[S][s] * F_inv[R][r] - F_inv[S][r] * F_inv[R][s] ) * eta_f;
								 d2_omega(i_grad_u_dot + 3 * r + R, i_F + 3 * s + S) = J/V_m_f * (F_inv[S][s] * F_inv[R][r] - F_inv[S][r] * F_inv[R][s] ) * eta_f;
							}
						}
						d2_omega(i_eta_f, i_F + 3 * r + R) = - J/V_m_f * (trace(F_inv * (grad_v_dot_u_dot)) * F_inv[R][r] - (F_inv * (grad_v_dot_u_dot * F_inv))[R][r] );
					}
				}
			}

		}
		return false;

	}
};


/**
 * Class defining Lagrangian multiplier term for an incompressible fluid.
 *
 * \f$ \omega^\Sigma =	\dfrac{J}{V^\mathrm{f}_\mathrm{m}}\boldsymbol{F}^{-1} : \left[ \boldsymbol{N} \left( \dot{\boldsymbol{v}} - \dot{\boldsymbol{u}} \right) \right] \eta^\mathrm{f}\f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential<br>
 * and \f$\boldsymbol{N}\f$ the unit normal vector.<br>
 *
 * This is the interfacial counterpart to OmegaFluidIncompressibility01.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0]  			\f$\dot{v}_x\f$<br>
 * 															[1]				\f$\dot{v}_y\f$<br>
 * 															[2]				\f$\dot{v}_z\f$<br>
 *															[3]				\f$\dot{u}_y\f$<br>
 *															[4]				\f$\dot{u}_y\f$<br>
 *															[5]				\f$\dot{u}_z\f$<br>
 * 															[6]				\f$\eta^\mathrm{f}\f$<br>
 * 															[7] ... [15]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidIncompressibility02 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * Number of ionic species \f$V^\mathrm{f}_\mathrm{m}\f$
	 */
	const double
	V_m_f;

	/**
	 * orientation of normal vector (if false, the normal vector points from - to +, if true the other way round)
	 */
	const bool
	flip_normal;

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
	 * @param[in]		V_m_f					OmegaFluidIncompressibility02::V_m_f
	 *
	 * @param[in]		flip_normal				OmegaFluidIncompressibility02::flip_normal
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluidIncompressibility02(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>										domain_of_integration,
									const dealii::Quadrature<spacedim-1>											quadrature,
									GlobalDataIncrementalFE<spacedim>&												global_data,
									const double																	V_m_f,
									const bool																		flip_normal,
									const unsigned int																method,
									const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 3, 3, 1, 9, method, alpha, "OmegaFluidIncompressibility02"),
	V_m_f(V_m_f),
	flip_normal(flip_normal)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{

		dealii::Tensor<2,3> F, F_inv;

		const double eta_f = values[6];

		dealii::Tensor<1,3> v_dot, u_dot, v_dot_u_dot;
		for(unsigned int r = 0; r < 3; ++r)
		{
			v_dot[r] = values[r];
			u_dot[r] = values[3 + r];
			for(unsigned int S = 0; S < 3; ++S)
				F[r][S] = values[7 + 3 * r + S];
		}
		F_inv = invert(F);
		const double J = determinant(F);
		v_dot_u_dot = v_dot - u_dot;

		dealii::Tensor<1, 3> n__, n_;
		n__[0] = n[0];
		n__[1] = spacedim < 2 ? 0.0 : n[1];
		n__[2] = spacedim < 3 ? 0.0 : n[2];
		n_ = flip_normal ? -J * transpose(F_inv) * n__ : J * transpose(F_inv) * n__;

		if(get<0>(requested_quantities))
		{
			sigma = eta_f * v_dot_u_dot * n_ / V_m_f;
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[6] = v_dot_u_dot * n_ / V_m_f;
			for(unsigned int r = 0; r < 3; ++r)
			{
				d_sigma[r] =  eta_f * n_[r] / V_m_f;
				d_sigma[r + 3] =  -eta_f * n_[r] / V_m_f;
			}
		}

		if(get<2>(requested_quantities))
		{

			for(unsigned int r = 0; r < 3; ++r)
			{
				d2_sigma(6, r) = d2_sigma(r, 6) = n_[r] / V_m_f;
				d2_sigma(6, r + 3) = d2_sigma(r + 3, 6) = -n_[r] / V_m_f;
			}

			if(compute_dq)
			{
				dealii::Tensor<3,3> dn_dF;
				dealii::Tensor<2,3> v_dot_u_dot_dn_dF;
				for(unsigned int m = 0; m < 3; ++m)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							dn_dF[m][r][R] = F_inv[R][r] * n_[m] - F_inv[R][m] * n_[r];
				v_dot_u_dot_dn_dF = v_dot_u_dot * dn_dF;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_sigma(6, 7 + r * 3 + R) = v_dot_u_dot_dn_dF[r][R] / V_m_f;
						for(unsigned int k = 0; k < 3; ++k)
						{
							d2_sigma(k, 7 + r * 3 + R) = eta_f * dn_dF[k][r][R] / V_m_f;
							d2_sigma(k + 3, 7 + r * 3 + R) = -eta_f * dn_dF[k][r][R] / V_m_f;
						}
					}
				}
			}
		}

		return false;
	}
};

/**
 * Class defining Lagrangian multiplier term for an incompressible fluid.
 *
 * \f$ \omega^\Omega =	-\dfrac{J}{V^\mathrm{f}_\mathrm{m}}\boldsymbol{F}^{-1} : \nabla\left( \dot{\boldsymbol{v}} - \dot{\boldsymbol{u}} \right) \eta^\mathrm{f}\f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * and \f$\eta^\mathrm{f}\f$ is the fluid potential<br>
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  			\f$\dot{v}_{,xx}\f$<br>
 * 																				[1]				\f$\dot{v}_{,xy}\f$<br>
 * 																				[2]				\f$\dot{v}_{,xz}\f$<br>
 * 																				[3]  			\f$\dot{v}_{,yx}\f$<br>
 * 																				[4]				\f$\dot{v}_{,yy}\f$<br>
 * 																				[5]				\f$\dot{v}_{,yz}\f$<br>
 * 																				[6]  			\f$\dot{v}_{,zx}\f$<br>
 * 																				[7]				\f$\dot{v}_{,zy}\f$<br>
 * 																				[8]				\f$\dot{v}_{,zz}\f$<br>
 *																				[9]				\f$\dot{u}_{,xx}\f$<br>
 *																				[10]			\f$\dot{u}_{,xy}\f$<br>
 *																				[11]			\f$\dot{u}_{,xz}\f$<br>
 *																				[12]			\f$\dot{u}_{,yx}\f$<br>
 *																				[13]			\f$\dot{u}_{,yy}\f$<br>
 *																				[14]			\f$\dot{u}_{,yz}\f$<br>
 *																				[15]			\f$\dot{u}_{,zx}\f$<br>
 *																				[16]			\f$\dot{u}_{,zy}\f$<br>
 *																				[17]			\f$\dot{u}_{,zz}\f$<br>
 * 																				[18]			\f$\eta^\mathrm{f}\f$<br>
 * 																				[19] ... [20]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidIncompressibility03 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	/**
	 * Number of ionic species \f$V^\mathrm{f}_\mathrm{m}\f$
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
	 * @param[in]		V_m_f					OmegaFluidIncompressibility01::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluidIncompressibility03(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>									domain_of_integration,
									const dealii::Quadrature<spacedim>											quadrature,
									GlobalDataIncrementalFE<spacedim>&											global_data,
									const double																V_m_f,
									const unsigned int															method,
									const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 9, 9, 1, 9, method, alpha, "OmegaDualFluidIncompressibility03"),
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

		// start indices for respective quantities
		const unsigned int i_grad_v_dot = 0;
		const unsigned int i_grad_u_dot = 9;
		const unsigned int i_eta_f = 18;
		const unsigned int i_F = 19;

		dealii::Tensor<2,3> grad_v_dot;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_v_dot[r][R] = values[i_grad_v_dot + 3 * r + R];

		dealii::Tensor<2,3> grad_u_dot;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_u_dot[r][R] = values[i_grad_u_dot + 3 * r + R];

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		const double J = determinant(F);
		dealii::Tensor<2,3> F_inv;
		F_inv = invert(F);

		dealii::Tensor<2,3> grad_v_dot_u_dot = grad_v_dot - grad_u_dot;

		if(get<0>(requested_quantities))
		{
			omega = -J/V_m_f * trace(F_inv * (grad_v_dot_u_dot)) * eta_f;
		}

		if(get<1>(requested_quantities))
		{
			d_omega[i_eta_f]   = -J/V_m_f * trace(F_inv * (grad_v_dot_u_dot));
			for(unsigned int r = 0; r < 3; ++r)
			{
				for(unsigned int R = 0; R < 3; ++R)
				{
					d_omega[i_grad_v_dot + 3 * r + R] = -J / V_m_f * F_inv[R][r] * eta_f;
					d_omega[i_grad_u_dot + 3 * r + R] = J / V_m_f * F_inv[R][r] * eta_f;
				}
			}
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
			{
				for(unsigned int R = 0; R < 3; ++R)
				{
					d2_omega(i_eta_f, i_grad_v_dot + 3 * r + R) = d2_omega(i_grad_v_dot + 3 * r + R, i_eta_f) = -J / V_m_f * F_inv[R][r];
					d2_omega(i_eta_f, i_grad_u_dot + 3 * r + R) = d2_omega(i_grad_u_dot + 3 * r + R, i_eta_f) = J / V_m_f * F_inv[R][r];
				}
			}


			if(compute_dq)
			{
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						for(unsigned int s = 0; s < 3; ++s)
						{
							for(unsigned int S = 0; S < 3; ++S)
							{
								 d2_omega(i_grad_v_dot + 3 * r + R, i_F + 3 * s + S) = -J/V_m_f * (F_inv[S][s] * F_inv[R][r] - F_inv[S][r] * F_inv[R][s] ) * eta_f;
								 d2_omega(i_grad_u_dot + 3 * r + R, i_F + 3 * s + S) = J/V_m_f * (F_inv[S][s] * F_inv[R][r] - F_inv[S][r] * F_inv[R][s] ) * eta_f;
							}
						}
						d2_omega(i_eta_f, i_F + 3 * r + R) = - J/V_m_f * (trace(F_inv * (grad_v_dot_u_dot)) * F_inv[R][r] - (F_inv * (grad_v_dot_u_dot * F_inv))[R][r] );
					}
				}
			}

		}
		return false;

	}
};

/**
 * Class defining Lagrangian multiplier term for an incompressible fluid.
 *
 * \f$ \omega^\Sigma =	\dfrac{J}{V^\mathrm{f}_\mathrm{m}}\boldsymbol{F}^{-1} : \left[ \boldsymbol{N} \left( \dot{\boldsymbol{v}} - \dot{\boldsymbol{u}} \right) - \dot{I} \right] \eta^\mathrm{f}\f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\dot{I}\f$ a normal flux,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * \f$\boldsymbol{u}\f$ the displacement variable,<br>
 * \f$c^\mathrm{f}\f$ the fluid concentration,<br>
 * \f$\eta^\mathrm{f}\f$ is the fluid potential<br>
 * and \f$\boldsymbol{N}\f$ the unit normal vector.<br>
 *
 * This is the interfacial counterpart to OmegaFluidIncompressibility01.
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0]  			\f$\dot{v}_x\f$<br>
 * 															[1]				\f$\dot{v}_y\f$<br>
 * 															[2]				\f$\dot{v}_z\f$<br>
 *															[3]				\f$\dot{u}_y\f$<br>
 *															[4]				\f$\dot{u}_y\f$<br>
 *															[5]				\f$\dot{u}_z\f$<br>
 *															[6]				\f$\dot{I}\f$<br>
 * 															[7]				\f$\eta^\mathrm{f}\f$<br>
 * 															[8] ... [16]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidIncompressibility04 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * Number of ionic species \f$V^\mathrm{f}_\mathrm{m}\f$
	 */
	const double
	V_m_f;

	/**
	 * orientation of normal vector (if false, the normal vector points from - to +, if true the other way round)
	 */
	const bool
	flip_normal;

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
	 * @param[in]		V_m_f					OmegaFluidIncompressibility02::V_m_f
	 *
	 * @param[in]		flip_normal				OmegaFluidIncompressibility02::flip_normal
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluidIncompressibility04(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>										domain_of_integration,
									const dealii::Quadrature<spacedim-1>											quadrature,
									GlobalDataIncrementalFE<spacedim>&												global_data,
									const double																	V_m_f,
									const bool																		flip_normal,
									const unsigned int																method,
									const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 7, 0, 1, 9, method, alpha, "OmegaFluidIncompressibility04"),
	V_m_f(V_m_f),
	flip_normal(flip_normal)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{

		dealii::Tensor<2,3> F, F_inv;

		const double I_dot = values[6];
		const double eta_f = values[7];

		dealii::Tensor<1,3> v_dot, u_dot, v_dot_u_dot;
		for(unsigned int r = 0; r < 3; ++r)
		{
			v_dot[r] = values[r];
			u_dot[r] = values[3 + r];
			for(unsigned int S = 0; S < 3; ++S)
				F[r][S] = values[8 + 3 * r + S];
		}
		F_inv = invert(F);
		const double J = determinant(F);
		v_dot_u_dot = v_dot - u_dot;

		dealii::Tensor<1, 3> n__, n_;
		n__[0] = n[0];
		n__[1] = spacedim < 2 ? 0.0 : n[1];
		n__[2] = spacedim < 3 ? 0.0 : n[2];
		n_ = flip_normal ? -J * transpose(F_inv) * n__ : J * transpose(F_inv) * n__;

		if(get<0>(requested_quantities))
		{
			sigma = eta_f * (v_dot_u_dot * n_ / V_m_f - I_dot);
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[6] = -eta_f;
			d_sigma[7] = v_dot_u_dot * n_ / V_m_f - I_dot;
			for(unsigned int r = 0; r < 3; ++r)
			{
				d_sigma[r] =  eta_f * n_[r] / V_m_f;
				d_sigma[r + 3] =  -eta_f * n_[r] / V_m_f;
			}
		}

		if(get<2>(requested_quantities))
		{

			for(unsigned int r = 0; r < 3; ++r)
			{
				d2_sigma(6, 7) = d2_sigma(7, 6) = -1.0;
				d2_sigma(7, r) = d2_sigma(r, 7) = n_[r] / V_m_f;
				d2_sigma(7, r + 3) = d2_sigma(r + 3, 7) = -n_[r] / V_m_f;
			}

			if(compute_dq)
			{
				dealii::Tensor<3,3> dn_dF;
				dealii::Tensor<2,3> v_dot_u_dot_dn_dF;
				for(unsigned int m = 0; m < 3; ++m)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							dn_dF[m][r][R] = F_inv[R][r] * n_[m] - F_inv[R][m] * n_[r];
				v_dot_u_dot_dn_dF = v_dot_u_dot * dn_dF;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_sigma(7, 8 + r * 3 + R) = v_dot_u_dot_dn_dF[r][R] / V_m_f;
						for(unsigned int k = 0; k < 3; ++k)
						{
							d2_sigma(k, 8 + r * 3 + R) = eta_f * dn_dF[k][r][R] / V_m_f;
							d2_sigma(k + 3, 8 + r * 3 + R) = -eta_f * dn_dF[k][r][R] / V_m_f;
						}
					}
				}
			}
		}

		return false;
	}
};

/**
 * Class defining Lagrangian multiplier term for an incompressible fluid.
 *
 * \f$ \omega^\Sigma =	-\dfrac{J}{V^\mathrm{f}_\mathrm{m}}\boldsymbol{F}^{-1} : \nabla \dot{\boldsymbol{v}} \, \eta \f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ the determinant of the deformation gradient,<br>
 * \f$V^\mathrm{f}_\mathrm{m}\f$ the molar volume of the fluid,<br>
 * and \f$\eta\f$ a Lagrange multiplier.<br>
 *
 * This enforces incompressibility as an interface condition
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\dot{v}_{,xx}\f$<br>
 * 															[1]	\f$\dot{v}_{,xy}\f$<br>
 * 															[2]	\f$\dot{v}_{,xz}\f$<br>
 * 															[3] \f$\dot{v}_{,yx}\f$<br>
 * 															[4]	\f$\dot{v}_{,yy}\f$<br>
 * 															[5]	\f$\dot{v}_{,yz}\f$<br>
 * 															[6] \f$\dot{v}_{,zx}\f$<br>
 * 															[7]	\f$\dot{v}_{,zy}\f$<br>
 * 															[8]	\f$\dot{v}_{,zz}\f$<br>
 * 															[9]	\f$\eta\f$<br>
 * 															[10] ... [18]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidIncompressibility05 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * Number of ionic species \f$V^\mathrm{f}_\mathrm{m}\f$
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
	 * @param[in]		V_m_f					OmegaFluidIncompressibility05::V_m_f
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaFluidIncompressibility05(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>										domain_of_integration,
									const dealii::Quadrature<spacedim-1>											quadrature,
									GlobalDataIncrementalFE<spacedim>&												global_data,
									const double																	V_m_f,
									const unsigned int																method,
									const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 9, 0, 1, 9, method, alpha, "OmegaFluidIncompressibility05"),
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
								const dealii::Tensor<1, spacedim>&	/*n*/,
								double&								sigma,
								dealii::Vector<double>&				d_sigma,
								dealii::FullMatrix<double>&			d2_sigma,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{


		// start indices for respective quantities
		const unsigned int i_grad_v_dot = 0;
		const unsigned int i_eta_f = 9;
		const unsigned int i_F = 10;

		dealii::Tensor<2,3> grad_v_dot;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_v_dot[r][R] = values[i_grad_v_dot + 3 * r + R];

		const double eta_f = values[i_eta_f];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		const double J = determinant(F);
		dealii::Tensor<2,3> F_inv;
		F_inv = invert(F);

		if(get<0>(requested_quantities))
		{
			sigma = -J/V_m_f * trace(F_inv * grad_v_dot) * eta_f;
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[i_eta_f]   = -J/V_m_f * trace(F_inv * grad_v_dot);
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int R = 0; R < 3; ++R)
					d_sigma[i_grad_v_dot + 3 * r + R] = -J / V_m_f * F_inv[R][r] * eta_f;
		}

		if(get<2>(requested_quantities))
		{

			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int R = 0; R < 3; ++R)
					d2_sigma(i_eta_f, i_grad_v_dot + 3 * r + R) = d2_sigma(i_grad_v_dot + 3 * r + R, i_eta_f) = -J / V_m_f * F_inv[R][r];


			if(compute_dq)
			{
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						for(unsigned int s = 0; s < 3; ++s)
							for(unsigned int S = 0; S < 3; ++S)
								 d2_sigma(i_grad_v_dot + 3 * r + R, i_F + 3 * s + S) = -J/V_m_f * (F_inv[S][s] * F_inv[R][r] - F_inv[S][r] * F_inv[R][s] ) * eta_f;
						d2_sigma(i_eta_f, i_F + 3 * r + R) = - J/V_m_f * (trace(F_inv * (grad_v_dot)) * F_inv[R][r] - (F_inv * (grad_v_dot * F_inv))[R][r] );
					}
				}
			}
		}

		return false;
	}
};


/**
 * Class defining a normal boundary condition of the form.
 *
 * \f$ \omega^\Sigma =	\eta (\boldsymbol{N} \cdot \boldsymbol{F}^{-1}) \cdot \mathrm{sym}\left( \nabla \dot{\boldsymbol{v}} \cdot \boldsymbol{F}^{-1} + \dot{\xi} \boldsymbol{I} \right) \cdot (\boldsymbol{N} \cdot \boldsymbol{F}^{-1}) \f$,
 *
 * where \f$\dot{\boldsymbol{v}}\f$ is the fluid velocity,<br>
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$\dot{\xi}\f$ a scalar process variable,<br>
 * and \f$\eta\f$ a Lagrange multiplier.<br>
 *
 * This enforces incompressibility as an interface condition
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\dot{v}_{,xx}\f$<br>
 * 															[1]	\f$\dot{v}_{,xy}\f$<br>
 * 															[2]	\f$\dot{v}_{,xz}\f$<br>
 * 															[3] \f$\dot{v}_{,yx}\f$<br>
 * 															[4]	\f$\dot{v}_{,yy}\f$<br>
 * 															[5]	\f$\dot{v}_{,yz}\f$<br>
 * 															[6] \f$\dot{v}_{,zx}\f$<br>
 * 															[7]	\f$\dot{v}_{,zy}\f$<br>
 * 															[8]	\f$\dot{v}_{,zz}\f$<br>
 * 															[9]	\f$\dot{\xi}\f$<br>
 * 															[10] \f$\eta\f$<br>
 * 															[11] ... [19]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidNormalCondition00 : public incrementalFE::Omega<spacedim-1, spacedim>
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
	OmegaFluidNormalCondition00(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>										domain_of_integration,
								const dealii::Quadrature<spacedim-1>											quadrature,
								GlobalDataIncrementalFE<spacedim>&												global_data,
								const unsigned int																method,
								const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 10, 0, 1, 9, method, alpha, "OmegaFluidNormalCondition00")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{


		// start indices for respective quantities
		const unsigned int i_grad_v_dot = 0;
		const unsigned int i_xi_dot = 9;
		const unsigned int i_eta = 10;
		const unsigned int i_F = 11;

		dealii::Tensor<2,3> grad_v_dot_ref, grad_v_dot, d;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_v_dot_ref[r][R] = values[i_grad_v_dot + 3 * r + R];

		const double xi_dot = values[i_xi_dot];

		const double eta = values[i_eta];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		dealii::Tensor<2,3> F_inv;
		F_inv = invert(F);

		grad_v_dot = grad_v_dot_ref * F_inv;

		dealii::Tensor<1, 3> n__, n_tilde, F_inv_n_tilde;
		n__[0] = n[0];
		n__[1] = spacedim < 2 ? 0.0 : n[1];
		n__[2] = spacedim < 3 ? 0.0 : n[2];
		n_tilde = transpose(F_inv) * n__;

		F_inv_n_tilde = F_inv * n_tilde;

		d = 2.0 * symmetrize(grad_v_dot);

		for(unsigned int m = 0; m < 3; ++m)
			d[m][m] += xi_dot;

		if(get<0>(requested_quantities))
		{
			sigma = eta * (n_tilde * d) * n_tilde;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int i = 0; i < 3; ++i)
				for(unsigned int J = 0; J < 3; ++J)
					d_sigma[i_grad_v_dot + 3 * i + J] = 2.0 * eta * n_tilde[i] * F_inv_n_tilde[J];
			d_sigma[i_xi_dot]   = eta * (n_tilde * n_tilde);
			d_sigma[i_eta]   = (n_tilde * d) * n_tilde;
		}

		if(get<2>(requested_quantities))
		{

			d2_sigma(i_xi_dot, i_eta) = d2_sigma(i_eta, i_xi_dot) = n_tilde * n_tilde;
			for(unsigned int i = 0; i < 3; ++i)
				for(unsigned int J = 0; J < 3; ++J)
					d2_sigma(i_eta, i_grad_v_dot + 3 * i + J) = d2_sigma(i_grad_v_dot + 3 * i + J, i_eta) = 2.0 * n_tilde[i] * F_inv_n_tilde[J];

			if(compute_dq)
			{
				dealii::Tensor<3,3> dn_tilde_dF, F_inv_dn_tilde_dF;
				for(unsigned int k = 0; k < 3; ++k)
					for(unsigned int i = 0; i < 3; ++i)
						for(unsigned int J = 0; J < 3; ++J)
							dn_tilde_dF[k][i][J] = -n_tilde[i] * F_inv[J][k];
				F_inv_dn_tilde_dF = F_inv * dn_tilde_dF;
				dealii::Tensor<2,3> n_tilde_dn_tilde_dF = n_tilde * dn_tilde_dF;

				for(unsigned int i = 0; i < 3; ++i)
					for(unsigned int J = 0; J < 3; ++J)
						for(unsigned int r = 0; r < 3; ++r)
							for(unsigned int R = 0; R < 3; ++R)
								d2_sigma(i_grad_v_dot + 3 * i + J, i_F + 3 * r + R) = 2.0 * eta * ( dn_tilde_dF[i][r][R] * F_inv_n_tilde[J] + F_inv_dn_tilde_dF[J][r][R] * n_tilde[i] - n_tilde[i] * F_inv[J][r] * F_inv_n_tilde[R] );
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_sigma(i_xi_dot, i_F + 3 * r + R) = 2.0 * eta * n_tilde_dn_tilde_dF[r][R];
						for(unsigned int k = 0; k < 3; ++k)
							for(unsigned int l = 0; l < 3; ++l)
								d2_sigma(i_eta, i_F + 3 * r + R) += 2.0 * ( dn_tilde_dF[l][r][R] * d[k][l] * n_tilde[k] - n_tilde[k] * grad_v_dot[k][r] * F_inv[R][l] * n_tilde[l]);
					}
				}
			}
		}

		return false;
	}
};

/**
 * Class defining a normal boundary condition of the form.
 *
 * \f$ \omega^\Sigma =	- \boldsymbol{n}\cdot\boldsymbol{d}\cdot\boldsymbol{n} \, J \boldsymbol{f} \cdot \boldsymbol{F}^{-\top} \cdot \boldsymbol{N} \f$,
 *
 * where \f$ \boldsymbol{d} = \mathrm{sym}\left( \nabla \dot{\boldsymbol{v}} \cdot \boldsymbol{F}^{-1} \right) \f$,
 * \f$ \boldsymbol{n} = J\,\boldsymbol{F}^{-\top} \cdot \boldsymbol{N} / \sqrt{ \boldsymbol{N} \cdot \boldsymbol{F}^{-\top} \cdot \boldsymbol{F} \cdot \boldsymbol{N} } \f$
 * \f$\boldsymbol{F}\f$ the deformation gradient,<br>
 * \f$J\f$ its determinant,
 * \f$\dot{\boldsymbol{v}}\f$ a process variable,
 * \f$\boldsymbol{f}\f$ a Lagrangian multiplier.
 *
 * This enforces incompressibility as an interface condition
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :<br>[0] \f$\dot{v}_{,xx}\f$<br>
 * 															[1]	\f$\dot{v}_{,xy}\f$<br>
 * 															[2]	\f$\dot{v}_{,xz}\f$<br>
 * 															[3] \f$\dot{v}_{,yx}\f$<br>
 * 															[4]	\f$\dot{v}_{,yy}\f$<br>
 * 															[5]	\f$\dot{v}_{,yz}\f$<br>
 * 															[6] \f$\dot{v}_{,zx}\f$<br>
 * 															[7]	\f$\dot{v}_{,zy}\f$<br>
 * 															[8]	\f$\dot{v}_{,zz}\f$<br>
 * 															[9]	\f$f_x\f$<br>
 * 															[10] \f$f_y\f$<br>
 * 															[11] \f$f_z\f$<br>
 * 															[12] ... [20]	\f$F_{xx}, F_{xy}, F_{xz}, F_{yx}, F_{yy}, F_{yz}, F_{zx}, F_{zy}, F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaFluidNormalCondition01 : public incrementalFE::Omega<spacedim-1, spacedim>
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
	OmegaFluidNormalCondition01(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
								const std::set<dealii::types::material_id>										domain_of_integration,
								const dealii::Quadrature<spacedim-1>											quadrature,
								GlobalDataIncrementalFE<spacedim>&												global_data,
								const unsigned int																method,
								const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 9, 0, 3, 9, method, alpha, "OmegaFluidNormalCondition01")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{


		// start indices for respective quantities
		const unsigned int i_grad_v_dot = 0;
		const unsigned int i_f = 9;
		const unsigned int i_F = 12;

		dealii::Tensor<2,3> grad_v_dot_ref, grad_v_dot, d;
		for(unsigned int r = 0; r < 3; ++r)
			for(unsigned int R = 0; R < 3; ++R)
				grad_v_dot_ref[r][R] = values[i_grad_v_dot + 3 * r + R];

		dealii::Tensor<1,3> f;
		for(unsigned int r = 0; r < 3; ++r)
			f[r] = values[i_f + r];

		dealii::Tensor<2,3> F;
		for(unsigned int m = 0; m < 3; ++m)
			for(unsigned int n = 0; n < 3; ++n)
				F[m][n] = values[i_F + m * 3 + n];
		dealii::Tensor<2,3> F_inv, C_inv;
		F_inv = invert(F);
		C_inv = F_inv * transpose(F_inv);
		const double J = determinant(F);

		grad_v_dot = grad_v_dot_ref * F_inv;

		dealii::Tensor<1,3> N, n_, n_tilde;
		N[0] = n[0];
		N[1] = spacedim < 2 ? 0.0 : n[1];
		N[2] = spacedim < 3 ? 0.0 : n[2];

		const double h = sqrt(N * C_inv * N);
		n_tilde = J * N * F_inv;
		n_ = n_tilde / h;

		dealii::Tensor<1,3> F_inv_n = F_inv * n_;
		const double n_tilde_f = n_tilde * f;

		if(get<0>(requested_quantities))
		{
			sigma = - 2.0 * (n_ * grad_v_dot * n_) * (n_tilde * f);
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int N = 0; N < 3; ++N)
					d_sigma[i_grad_v_dot + 3 * m + N] = -2.0 * n_[m] * F_inv_n[N] * n_tilde_f;
			for(unsigned int i = 0; i < 3; ++i)
				d_sigma[i_f + i] = -2.0 * (n_ * grad_v_dot * n_) * n_tilde[i];
		}

		if(get<2>(requested_quantities))
		{

			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int N = 0; N < 3; ++N)
					for(unsigned int i = 0; i < 3; ++i)
						d2_sigma(i_grad_v_dot + 3 * m + N, i_f + i) = d2_sigma(i_f + i, i_grad_v_dot + 3 * m + N) = -2.0 * n_[m] * F_inv_n[N] * n_tilde[i];

			if(compute_dq)
			{
				dealii::Tensor<2,3> dh_dF;
				for(unsigned int r = 0; r < 3; ++r)
					for(unsigned int R = 0; R < 3; ++R)
						dh_dF[r][R] = -1.0 / J * n_[r] * (N * C_inv)[R];
				dealii::Tensor<3,3> dn_dF, F_inv_dn_dF;
				for(unsigned int k = 0; k < 3; ++k)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							dn_dF[k][r][R] = F_inv[R][r] * n_[k] - F_inv[R][k] * n_[r] - n_[k] / h * dh_dF[r][R];
				F_inv_dn_dF = F_inv * dn_dF;
				dealii::Tensor<1,3> F_inv_f = F_inv * f;

//				dealii::Tensor<2,3> n_tilde_dn_tilde_dF = n_tilde * dn_tilde_dF;

				for(unsigned int m = 0; m < 3; ++m)
					for(unsigned int N = 0; N < 3; ++N)
						for(unsigned int r = 0; r < 3; ++r)
							for(unsigned int R = 0; R < 3; ++R)
								d2_sigma(i_grad_v_dot + 3 * m + N, i_F + 3 * r + R) = 	-2.0 * dn_dF[m][r][R] * F_inv_n[N] * n_tilde_f
																						- 2.0 * n_[m] * F_inv_dn_dF[N][r][R] * n_tilde_f
																						- 2.0 * n_[m] * F_inv_n[N] *(F_inv[R][r] * n_tilde_f - F_inv_f[R] * n_tilde[r])
																						+ 2.0 * n_[m] * ( F_inv[N][r] * F_inv_n[R] * n_tilde_f);
				for(unsigned int i = 0; i < 3; ++i)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							for(unsigned int k = 0; k < 3; ++k)
								for(unsigned int l = 0; l < 3; ++l)
									d2_sigma(i_f + i, i_F + 3 * r + R) += -2.0 * dn_dF[k][r][R] * grad_v_dot[k][l] * n_[l] * n_tilde[i]
																		  -2.0 * n_[k] * grad_v_dot[k][l] * dn_dF[l][r][R] * n_tilde[i]
																		  -2.0 * n_[k] * grad_v_dot[k][l] * n_[l] * ( F_inv[R][r] * n_tilde[i] - F_inv[R][i] * n_tilde[r])
																		  +2.0 * n_[k] * grad_v_dot[k][r] * F_inv[R][l] * n_[l] * n_tilde[i];
			}
		}

		return false;
	}
};



/**
 * Class defining a domain related scalar functional with the integrand
 *
 * \f$ \omega^\Omega =	2 J \boldsymbol{g}:\boldsymbol{d} \f$,
 *
 * where \f$\boldsymbol{d} = \mathrm{sym}\left(\nabla \boldsymbol{v} \cdot \boldsymbol{F}^{-1}\right)\f$ is the stretching,
 * \f$\boldsymbol{g} = \nabla \boldsymbol{f} \cdot \boldsymbol{F}^{-1}\f$ a corresponding Lagrangian multiplier term,
 * \f$\boldsymbol{F}\f$ the deformation gradient, and \f$J = \det\boldsymbol{F}\f$.
 *
 * This enforces the equilibrium applicable to a viscous fluid by means of a Lagrangian multiplier.
 *
 * Ordering of quantities in ScalarFunctional<spacedim, spacedim>::e_omega :<br>[0]  \f$v_{x,x}\f$<br>
 * 																				[1]  \f$v_{x,y}\f$<br>
 * 																				[2]  \f$v_{x,z}\f$<br>
 * 																				[3]  \f$v_{y,x}\f$<br>
 * 																				[4]  \f$v_{y,y}\f$<br>
 * 																				[5]  \f$v_{y,z}\f$<br>
 * 																				[6]  \f$v_{z,x}\f$<br>
 * 																				[7]  \f$v_{z,y}\f$<br>
 * 																				[8]  \f$v_{z,z}\f$<br>
 * 																				[9]  \f$f_{x,x}\f$<br>
 * 																				[10] \f$f_{x,y}\f$<br>
 * 																				[11] \f$f_{x,z}\f$<br>
 * 																				[12] \f$f_{y,x}\f$<br>
 * 																				[13] \f$f_{y,y}\f$<br>
 * 																				[14] \f$f_{y,z}\f$<br>
 * 																				[15] \f$f_{z,x}\f$<br>
 * 																				[16] \f$f_{z,y}\f$<br>
 * 																				[17] \f$f_{z,z}\f$<br>
 * 																				[18] \f$F_{xx}\f$<br>
 * 																				[19] \f$F_{xy}\f$<br>
 * 																				[20] \f$F_{xz}\f$<br>
 * 																				[21] \f$F_{yx}\f$<br>
 * 																				[22] \f$F_{yy}\f$<br>
 * 																				[23] \f$F_{yz}\f$<br>
 * 																				[24] \f$F_{zx}\f$<br>
 * 																				[25] \f$F_{zy}\f$<br>
 * 																				[26] \f$F_{zz}\f$
 */
template<unsigned int spacedim>
class OmegaLagrangeViscousDissipation00 : public incrementalFE::Omega<spacedim, spacedim>
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
	OmegaLagrangeViscousDissipation00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
										const std::set<dealii::types::material_id>									domain_of_integration,
										const dealii::Quadrature<spacedim>											quadrature,
										GlobalDataIncrementalFE<spacedim>&											global_data,
										const unsigned int															method,
										const double																alpha = 0.0)
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 9, 0, 9, 9, method, alpha, "OmegaLagrangeViscousDissipation00")
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

		dealii::Tensor<2,3> grad_ref_v, grad_ref_f, F, F_inv, C_inv, grad_v, grad_f, grad_v_sym, grad_f_sym, delta;
		for(unsigned int r = 0; r < 3; ++r)
		{
			for(unsigned int S = 0; S < 3; ++S)
			{
				grad_ref_v[r][S] = values[3 * r + S];
				grad_ref_f[r][S] = values[9 + 3 * r + S];
				F[r][S] = values[18 + 3 * r + S];
			}
		}
		F_inv = invert(F);
		C_inv = F_inv * transpose(F_inv);
		grad_v = grad_ref_v * F_inv;
		grad_v_sym = symmetrize(grad_v);
		grad_f = grad_ref_f * F_inv;
		grad_f_sym = symmetrize(grad_f);
		delta[0][0] = delta[1][1] = delta[2][2] = 1.0;
		const double J = determinant(F);

		if(get<0>(requested_quantities))
		{
			omega = 2.0 * J * trace( grad_f * symmetrize(grad_v) );
		}

		dealii::Tensor<2,3> T_f, T_v;
		if(get<1>(requested_quantities) || get<2>(requested_quantities))
		{
			T_f = 2.0 * J * grad_f_sym * transpose(F_inv);
			T_v = 2.0 * J * grad_v_sym * transpose(F_inv);
			if(get<1>(requested_quantities))
			{
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int S = 0; S < 3; ++S)
						d_omega[r * 3 + S] = T_f[r][S];
					for(unsigned int S = 0; S < 3; ++S)
						d_omega[9 + r * 3 + S] = T_v[r][S];
				}
			}
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
				for(unsigned int S = 0; S < 3; ++S)
					for(unsigned int p = 0; p < 3; ++p)
						for(unsigned int Q = 0; Q < 3; ++Q)
							d2_omega(r * 3 + S, 9 + p * 3 + Q) = d2_omega(9 + p * 3 + Q, r * 3 + S) = J * (delta[r][p] * C_inv[S][Q] + F_inv[Q][r] * F_inv[S][p]);
			if(compute_dq)
			{
				dealii::Tensor<2,3> F_inv_grad_v, F_inv_grad_v_T, F_inv_grad_f, F_inv_grad_f_T;
				F_inv_grad_v = F_inv * grad_v;
				F_inv_grad_v_T = F_inv * transpose(grad_v);
				F_inv_grad_f = F_inv * grad_f;
				F_inv_grad_f_T = F_inv * transpose(grad_f);

				for(unsigned int i = 0; i < 3; ++i)
				{
					for(unsigned int I = 0; I < 3; ++I)
					{
						for(unsigned int r = 0; r < 3; ++r)
						{
							for(unsigned int R = 0; R < 3; ++R)
							{
								d2_omega(i * 3 + I, r * 3 + R + 18) = F_inv[R][r] * T_f[i][I] - J * ( F_inv_grad_f[I][r] * F_inv[R][i] + C_inv[R][I] * grad_f[i][r] + ( F_inv_grad_f[R][i] + F_inv_grad_f_T[R][i] ) * F_inv[I][r] );
								d2_omega(9 + i * 3 + I, r * 3 + R + 18) = F_inv[R][r] * T_v[i][I] - J * ( F_inv_grad_v[I][r] * F_inv[R][i] + C_inv[R][I] * grad_v[i][r] + ( F_inv_grad_v[R][i] + F_inv_grad_v_T[R][i] ) * F_inv[I][r] );
							}
						}
					}
				}
			}
		}

		return false;
	}
};


/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	\boldsymbol{t} \cdot \left[ \left(\dot{\boldsymbol{u}} - \dot{\boldsymbol{v}}\right) \times \boldsymbol{n} \right] + \dfrac{1}{2} \left(\boldsymbol{t} \cdot \boldsymbol{n} \right)^2 \f$,
 *
 * where \f$\dot{\boldsymbol{u}}\f$ is the velocity of a solid at an interface to a fluid, \f$\dot{\boldsymbol{v}}\f$ is the velocity of the fluid at the interface,
 * \f$\boldsymbol{t}\f$ is a Lagrangian multiplier, \f$\boldsymbol{n} = J \boldsymbol{F}^{-\top} \cdot \boldsymbol{N}\f$ is the normal vector on the interface in the deformed configuration,
 * with \f$\boldsymbol{F}\f$ being the deformation gradient, \f$J\f$ its determinant, and \f$\boldsymbol{N}\f$ the normal vector on the interface in the reference configuration.
 *
 * This constrains the tangential relative velocity between a fluid and a solid surface to zero. In order to regularize the problem, a positive definite term is added for the normal component of \f$\boldsymbol{t}\f$.
 *
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :					<br>[0]  \f$\dot{u}_{x}\f$<br>
 * 																				[1]  \f$\dot{u}_{y}\f$<br>
 * 																				[2]  \f$\dot{u}_{z}\f$<br>
 * 																				[3]  \f$\dot{v}_{x}\f$<br>
 * 																				[4]  \f$\dot{v}_{y}\f$<br>
 * 																				[5]  \f$\dot{v}_{z}\f$<br>
 * 																				[6]  \f$t_x\f$<br>
 * 																				[7]  \f$t_y\f$<br>
 * 																				[8]  \f$t_z\f$<br>
 * 																				[9]  \f$F_{xx}\f$<br>
 * 																				[10] \f$F_{xy}\f$<br>
 * 																				[11] \f$F_{xz}\f$<br>
 * 																				[12] \f$F_{yx}\f$<br>
 * 																				[13] \f$F_{yy}\f$<br>
 * 																				[14] \f$F_{yz}\f$<br>
 * 																				[15] \f$F_{zx}\f$<br>
 * 																				[16] \f$F_{zy}\f$<br>
 * 																				[17] \f$F_{zz}\f$<br>
 */
template<unsigned int spacedim>
class OmegaZeroTangentialFlux00 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * orientation of normal vector (if false, the normal vector points from - to +, if true the other way round)
	 */
	const bool
	flip_normal;

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
	 * @param[in]		flip_normal				OmegaZeroTangentialFlux00::flip_normal
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaZeroTangentialFlux00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>										domain_of_integration,
								const dealii::Quadrature<spacedim-1>											quadrature,
								GlobalDataIncrementalFE<spacedim>&												global_data,
								const bool																		flip_normal,
								const unsigned int																method,
								const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 6, 0, 3, 9, method, alpha, "OmegaZeroTangentialFlux00"),
	flip_normal(flip_normal)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{

		dealii::Tensor<1,3> u_dot, v_dot, u_dot_v_dot, t;
		dealii::Tensor<2,3> F, F_inv;


		for(unsigned int r = 0; r < 3; ++r)
		{
			u_dot[r] = values[r];
			v_dot[r] = values[3 + r];
			t[r] = values[6 + r];
			for(unsigned int S = 0; S < 3; ++S)
				F[r][S] = values[9 + 3 * r + S];
		}
		F_inv = invert(F);
		u_dot_v_dot = u_dot - v_dot;
		const double J = determinant(F);

		dealii::Tensor<1, 3> n__, n_;
		n__[0] = n[0];
		n__[1] = spacedim < 2 ? 0.0 : n[1];
		n__[2] = spacedim < 3 ? 0.0 : n[2];
		n_ = flip_normal ? -J * transpose(F_inv) * n__ : J * transpose(F_inv) * n__;

		if(get<0>(requested_quantities))
		{
			sigma = t * cross_product_3d(u_dot_v_dot, n_) + 0.5 * (t * n_) * (t * n_);
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
			{
				d_sigma[r] = cross_product_3d(n_, t)[r];
				d_sigma[3 + r] = -cross_product_3d(n_, t)[r];
				d_sigma[6 + r] = cross_product_3d(u_dot_v_dot, n_)[r] + (t * n_) * n_[r];
			}
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma(6 + 0, 0) = d2_sigma(0, 6 + 0) = 0.0;
			d2_sigma(6 + 1, 0) = d2_sigma(0, 6 + 1) = -n_[2];
			d2_sigma(6 + 2, 0) = d2_sigma(0, 6 + 2) = n_[1];
			d2_sigma(6 + 0, 1) = d2_sigma(1, 6 + 0) = n_[2];
			d2_sigma(6 + 1, 1) = d2_sigma(1, 6 + 1) = 0.0;
			d2_sigma(6 + 2, 1) = d2_sigma(1, 6 + 2) = -n_[0];
			d2_sigma(6 + 0, 2) = d2_sigma(2, 6 + 0) = -n_[1];
			d2_sigma(6 + 1, 2) = d2_sigma(2, 6 + 1) = n_[0];
			d2_sigma(6 + 2, 2) = d2_sigma(2, 6 + 2) = 0.0;

			d2_sigma(6 + 0, 3) = d2_sigma(3, 6 + 0) = 0.0;
			d2_sigma(6 + 1, 3) = d2_sigma(3, 6 + 1) = n_[2];
			d2_sigma(6 + 2, 3) = d2_sigma(3, 6 + 2) = -n_[1];
			d2_sigma(6 + 0, 4) = d2_sigma(4, 6 + 0) = -n_[2];
			d2_sigma(6 + 1, 4) = d2_sigma(4, 6 + 1) = 0.0;
			d2_sigma(6 + 2, 4) = d2_sigma(4, 6 + 2) = n_[0];
			d2_sigma(6 + 0, 5) = d2_sigma(5, 6 + 0) = n_[1];
			d2_sigma(6 + 1, 5) = d2_sigma(5, 6 + 1) = -n_[0];
			d2_sigma(6 + 2, 5) = d2_sigma(5, 6 + 2) = 0.0;

			for(unsigned int m = 0; m < 3; ++m)
				for(unsigned int l = 0; l < 3; ++l)
					d2_sigma(6 + m, 6 + l) = n_[m] * n_[l];

			if(compute_dq)
			{
				dealii::Tensor<3,3> dn_dF;
				dealii::Tensor<2,3> t_dn_dF;
				const double t_n = t * n_;
				for(unsigned int m = 0; m < 3; ++m)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							dn_dF[m][r][R] = F_inv[R][r] * n_[m] - F_inv[R][m] * n_[r];
				t_dn_dF = t * dn_dF;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						d2_sigma(0, 9 + 3 * r + R) = dn_dF[1][r][R] * t[2] - dn_dF[2][r][R] * t[1];
						d2_sigma(1, 9 + 3 * r + R) = dn_dF[2][r][R] * t[0] - dn_dF[0][r][R] * t[2];
						d2_sigma(2, 9 + 3 * r + R) = dn_dF[0][r][R] * t[1] - dn_dF[1][r][R] * t[0];
						d2_sigma(3, 9 + 3 * r + R) = -dn_dF[1][r][R] * t[2] + dn_dF[2][r][R] * t[1];
						d2_sigma(4, 9 + 3 * r + R) = -dn_dF[2][r][R] * t[0] + dn_dF[0][r][R] * t[2];
						d2_sigma(5, 9 + 3 * r + R) = -dn_dF[0][r][R] * t[1] + dn_dF[1][r][R] * t[0];
						d2_sigma(6, 9 + 3 * r + R) = -dn_dF[1][r][R] * u_dot_v_dot[2] + dn_dF[2][r][R] * u_dot_v_dot[1] + t_dn_dF[r][R] * n_[0] + t_n * dn_dF[0][r][R];
						d2_sigma(7, 9 + 3 * r + R) = -dn_dF[2][r][R] * u_dot_v_dot[0] + dn_dF[0][r][R] * u_dot_v_dot[2] + t_dn_dF[r][R] * n_[1] + t_n * dn_dF[1][r][R];
						d2_sigma(8, 9 + 3 * r + R) = -dn_dF[0][r][R] * u_dot_v_dot[1] + dn_dF[1][r][R] * u_dot_v_dot[0] + t_dn_dF[r][R] * n_[2] + t_n * dn_dF[2][r][R];
					}
				}
			}
		}

		return false;
	}
};

/**
 * Class defining an interface related scalar functional with the integrand
 *
 * \f$ \omega^\Sigma =	t  \left(\dot{\boldsymbol{u}} - \dot{\boldsymbol{v}}\right) \cdot \boldsymbol{n} \f$,
 *
 * where \f$\dot{\boldsymbol{u}}\f$ is the velocity of a solid at an interface to a fluid, \f$\dot{\boldsymbol{v}}\f$ is the velocity of the fluid at the interface,
 * \f$t\f$ is a Lagrangian multiplier, \f$\boldsymbol{n} = J \boldsymbol{F}^{-\top} \cdot \boldsymbol{N}\f$ is the normal vector on the interface in the deformed configuration,
 * with \f$\boldsymbol{F}\f$ being the deformation gradient, \f$J\f$ its determinant, and \f$\boldsymbol{N}\f$ the normal vector on the interface in the reference configuration.
 *
 * This constrains the normal relative velocity between a fluid and a solid surface to zero.
 *
 *
 *
 * Ordering of quantities in ScalarFunctional::e_sigma :					<br>[0]  \f$\dot{u}_{x}\f$<br>
 * 																				[1]  \f$\dot{u}_{y}\f$<br>
 * 																				[2]  \f$\dot{u}_{z}\f$<br>
 * 																				[3]  \f$\dot{v}_{x}\f$<br>
 * 																				[4]  \f$\dot{v}_{y}\f$<br>
 * 																				[5]  \f$\dot{v}_{z}\f$<br>
 * 																				[6]  \f$t\f$<br>
 * 																				[7]  \f$F_{xx}\f$<br>
 * 																				[8] \f$F_{xy}\f$<br>
 * 																				[9] \f$F_{xz}\f$<br>
 * 																				[10] \f$F_{yx}\f$<br>
 * 																				[11] \f$F_{yy}\f$<br>
 * 																				[12] \f$F_{yz}\f$<br>
 * 																				[13] \f$F_{zx}\f$<br>
 * 																				[14] \f$F_{zy}\f$<br>
 * 																				[15] \f$F_{zz}\f$<br>
 */
template<unsigned int spacedim>
class OmegaZeroNormalFlux01 : public incrementalFE::Omega<spacedim-1, spacedim>
{
private:

	/**
	 * orientation of normal vector (if false, the normal vector points from - to +, if true the other way round)
	 */
	const bool
	flip_normal;

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
	 * @param[in]		flip_normal				OmegaZeroNormalFlux01::flip_normal
	 *
	 * @param[in]		method					Omega<spacedim, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<spacedim, spacedim>::alpha
	 */
	OmegaZeroNormalFlux01(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
							const std::set<dealii::types::material_id>										domain_of_integration,
							const dealii::Quadrature<spacedim-1>											quadrature,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const bool																		flip_normal,
							const unsigned int																method,
							const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 6, 0, 1, 9, method, alpha, "OmegaZeroNormalFlux01"),
	flip_normal(flip_normal)
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{

		dealii::Tensor<1,3> u_dot, v_dot, u_dot_v_dot;
		dealii::Tensor<2,3> F, F_inv;

		for(unsigned int r = 0; r < 3; ++r)
		{
			u_dot[r] = values[r];
			v_dot[r] = values[3 + r];

			for(unsigned int S = 0; S < 3; ++S)
				F[r][S] = values[7 + 3 * r + S];
		}
		const double t = values[6];

		F_inv = invert(F);
		u_dot_v_dot = u_dot - v_dot;
		const double J = determinant(F);

		dealii::Tensor<1, 3> n__, n_;
		n__[0] = n[0];
		n__[1] = spacedim < 2 ? 0.0 : n[1];
		n__[2] = spacedim < 3 ? 0.0 : n[2];
		n_ = flip_normal ? -J * transpose(F_inv) * n__ : J * transpose(F_inv) * n__;

		if(get<0>(requested_quantities))
		{
			sigma = t * u_dot_v_dot * n_;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int r = 0; r < 3; ++r)
			{
				d_sigma[r] = t * n_[r];
				d_sigma[3 + r] = -t * n_[r];
			}
			d_sigma[6] = u_dot_v_dot * n_;
		}

		if(get<2>(requested_quantities))
		{

			for(unsigned int r = 0; r < 3; ++r)
			{
				d2_sigma(r,6) = d2_sigma(6,r) = n_[r];
				d2_sigma(3 + r,6) = d2_sigma(6,r + 3) = -n_[r];
			}

			if(compute_dq)
			{
				dealii::Tensor<3,3> dn_dF;
				dealii::Tensor<2,3> u_dot_v_dot_dn_dF;
				for(unsigned int m = 0; m < 3; ++m)
					for(unsigned int r = 0; r < 3; ++r)
						for(unsigned int R = 0; R < 3; ++R)
							dn_dF[m][r][R] = F_inv[R][r] * n_[m] - F_inv[R][m] * n_[r];
				u_dot_v_dot_dn_dF = u_dot_v_dot * dn_dF;
				for(unsigned int r = 0; r < 3; ++r)
				{
					for(unsigned int R = 0; R < 3; ++R)
					{
						for(unsigned int k = 0; k < 3; ++k)
						{
							d2_sigma(k, 7 + r * 3 + R) = t * dn_dF[k][r][R];
							d2_sigma(3 + k, 7 + r * 3 + R) = -t * dn_dF[k][r][R];
						}
						d2_sigma(6, 7 + r * 3 + R) = u_dot_v_dot_dn_dF[r][R];
					}
				}
			}
		}

		return false;
	}
};



/**
 * Class defining an interface related scalar functional  constraining the tangential relative velocity between a fluid and a solid surface to zero
 *
 * Ordering of quantities in ScalarFunctional<spacedim>::e_sigma :			<br>[0]  \f$\dot{u}_{x}\f$<br>
 * 																				[1]  \f$\dot{u}_{y}\f$<br>
 * 																				[2]  \f$v_{x}\f$<br>
 * 																				[3]  \f$v_{y}\f$<br>
 * 																				[4]  \f$\lambda\f$<br>
 * 																				[5]  \f$F_{xx}\f$<br>
 * 																				[6]  \f$F_{xy}\f$<br>
 * 																				[7]  \f$F_{yx}\f$<br>
 * 																				[8]  \f$F_{yy}\f$<br>
 */
template<unsigned int spacedim>
class OmegaZeroTangentialFlux2D00 : public incrementalFE::Omega<spacedim-1, spacedim>
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
	OmegaZeroTangentialFlux2D00(const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
								const std::set<dealii::types::material_id>										domain_of_integration,
								const dealii::Quadrature<spacedim-1>											quadrature,
								GlobalDataIncrementalFE<spacedim>&												global_data,
								const unsigned int																method,
								const double																	alpha = 0.0)
	:
	Omega<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 2, 2, 1, 4, method, alpha, "OmegaZeroTangentialFlux2D00")
	{
	}

	/**
	 * @see Omega<spacedim, spacedim>::get_values_and_derivatives()
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
								const bool							compute_dq)
	const
	{
		(void)compute_dq;
		Assert(!compute_dq, dealii::ExcMessage("The alpha-family for temporal discretization is not currently implemented!"));
		Assert(spacedim == 2, dealii::ExcMessage("This is currently only implemented in two dimensions!"));

		const double dv_0 = values[0] - values[2];
		const double dv_1 = values[1] - values[3];
		const double lambda = values[4];

		dealii::Tensor<1, spacedim> n_, t_;
		dealii::Tensor<2, spacedim> F;
		F[0][0] = values[5];
		F[0][1] = values[6];
		F[1][0] = values[7];
		F[1][1] = values[8];
		n_ = transpose(invert(F)) * n;
		n_ *= 1.0 / sqrt(n_ * n_);
		t_[0] = n_[1];
		t_[1] = -n_[0];

		if(get<0>(requested_quantities))
		{
			sigma = lambda * ( t_[0] * dv_0 + t_[1] * dv_1 );
		}

		if(get<1>(requested_quantities))
		{
			d_sigma[0] = lambda * t_[0];
			d_sigma[1] = lambda * t_[1];
			d_sigma[2] = -lambda * t_[0];
			d_sigma[3] = -lambda * t_[1];
			d_sigma[4] = t_[0] * dv_0 + t_[1] * dv_1;
		}

		if(get<2>(requested_quantities))
		{
			d2_sigma(0,4) = d2_sigma(4,0) = t_[0];
			d2_sigma(1,4) = d2_sigma(4,1) = t_[1];
			d2_sigma(2,4) = d2_sigma(4,2) = -t_[0];
			d2_sigma(3,4) = d2_sigma(4,3) = -t_[1];
		}

		return false;
	}
};

/**
 * Class defining the following functions
 *
 * \f$ \omega^\mathrm{C} = -\dot{\bar{J}}\varphi\f$ for prescribed current loading (loading type J),
 * \f$ \omega^\mathrm{C} = -\dot{J}(\varphi - \bar{\varphi})\f$ for prescribed voltage loading (loading type V),
 * \f$ \omega^\mathrm{C} = - \dfrac{1}{2 R^\mathrm{el}} \varphi^2\f$ for discharging through electrical resistance \f$R^\mathrm{el}\f$ (loading type R),
 *
 * The loading type can be changed at any time by changing OmegaElectricalLoading::loading_type.
 *
 * Similarly, the prescribed current, the prescribed voltage, and the electrical resistance can be changed at any time by assigning the respective variables
 * OmegaElectricalLoading::j_bar, OmegaElectricalLoading::phi, and OmegaElectricalLoading::R_el.
 *
 * Ordering of quantities in ScalarFunctional<0, spacedim>::C :<br>	[0]  \f$J\f$<br>
 * 																	[1]  \f$\varphi\f$
 */
template<unsigned int spacedim>
class OmegaElectricalLoading : public incrementalFE::Omega<0, spacedim>
{

public:

	/**
	 * electrical current if loading type J
	 */
	double
	j_bar = 0.0;

	/**
	 * prescribed voltage if loading type V
	 */
	double
	phi = 0.0;

	/**
	 * prescribed resistance if loading type R
	 */
	double
	R_el = 0.0;

	/**
	 * 0 : prescribed current		(loading type J)
	 * 1 : prescribed voltage		(loading type V)
	 * 2 : prescribed resistance	(loading type R)
	 */
	unsigned int
	loading_type = 0;

	/**
	 * constructor
	 *
	 * @param[in]		C						TotalPotentialContribution::C
	 *
	 * @param[in]		global_data				Omega<0, spacedim>::global_data
	 *
	 * @param[in]		method					Omega<0, spacedim>::method
	 *
	 * @param[in]		alpha					Omega<0, spacedim>::alpha
	 */
	OmegaElectricalLoading(	const std::vector<const dealii::GalerkinTools::IndependentField<0, spacedim>*>	C,
							GlobalDataIncrementalFE<spacedim>&												global_data,
							const unsigned int																method,
							const double																	alpha)
	:
	Omega<0, spacedim>(C, global_data, 1, 0, 1, 0, method, alpha, "OmegaElectricalLoading")
	{
	}

	bool get_values_and_derivatives(const dealii::Vector<double>& 		values,
									const double						/*t*/,
									double&								omega,
									dealii::Vector<double>&				d_omega,
									dealii::FullMatrix<double>&			d2_omega,
									const std::tuple<bool, bool, bool>	requested_quantities,
									const bool							/*compute_d2q*/)
	const
	{
		const double j_ap = values[0];
		const double phi_ap = values[1];

		if(loading_type == 0)
		{
			if(get<0>(requested_quantities))
				omega = -j_bar * phi_ap;

			if(get<1>(requested_quantities))
			{
				d_omega[0] = 0.0;
				d_omega[1] = -j_bar;
			}

			if(get<2>(requested_quantities))
			{
				d2_omega(0,0) = 0.0;
				d2_omega(1,1) = 0.0;
				d2_omega(0,1) = d2_omega(1,0) = 0.0;
			}
		}
		else if(loading_type == 1)
		{
			if(get<0>(requested_quantities))
				omega = -j_ap * (phi_ap - phi);

			if(get<1>(requested_quantities))
			{
				d_omega[0] = -(phi_ap - phi);
				d_omega[1] = -j_ap;
			}

			if(get<2>(requested_quantities))
			{
				d2_omega(0,0) = 0.0;
				d2_omega(1,1) = 0.0;
				d2_omega(0,1) = d2_omega(1,0) = -1.0;
			}
		}
		else if(loading_type == 2)
		{
			if(get<0>(requested_quantities))
				omega = -0.5 / R_el * phi_ap * phi_ap;

			if(get<1>(requested_quantities))
			{
				d_omega[0] = 0.0;
				d_omega[1] = -1.0 / R_el * phi_ap;
			}

			if(get<2>(requested_quantities))
			{
				d2_omega(0,0) = 0.0;
				d2_omega(1,1) = -1.0 / R_el;
				d2_omega(0,1) = d2_omega(1,0) = 0.0;
			}
		}
		else
		{
			Assert(false, dealii::ExcMessage("Unknown loading type!"));
		}

		return false;
	}
};

}


#endif /* INCREMENTALFE_SCALARFUNCTIONALS_OMEGALIB_H_ */
