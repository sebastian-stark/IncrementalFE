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

#ifndef INCREMENTALFE_CONSTRAINTS_H_
#define INCREMENTALFE_CONSTRAINTS_H_

#include <vector>
#include <atomic>

#include <incremental_fe/config.h>

#include <deal.II/base/subscriptor.h>

#include <galerkin_tools/dirichlet_constraint.h>

namespace incrementalFE
{

/**
 * Class collecting the constraints of the problem.
 *
 * The Constraints class inherits from Subscriptor in order to be
 * able to check that Constraints objects are only destroyed when they are
 * not needed anymore by other objects.
 *
 * @tparam	spacedim	spatial dimension of the problem
 */
template<unsigned int spacedim>
class Constraints : public dealii::Subscriptor
{

private:

	/**
	 * %Vector with the DirichletConstraint objects
	 *
	 * The first element of each pair is the constraint.
	 *
	 * The second entry of each pair defines the instant of time within a time step, at which the time dependent functions of the corresponding constraint (e.g. the constraint inhomogeneities) are evaluated.
	 * In particular, values in the range [0, 1] are admissible for each entry, where e.g. 0 means that the time dependent functions are evaluated at the time corresponding to the
	 * beginning of the time step and 1 means that they are evaluated at the time corresponding to the end of the time step.
	 *
	 * Typically, the constraints for state and process variables are evaluated at the end of the time step, while constraints for Lagrangian multipliers are evaluated at the instant of time corresponding to
	 * the time integration parameter \f$\alpha\f$.
	 */
	std::vector< std::pair<dealii::SmartPointer<const dealii::GalerkinTools::DirichletConstraint<spacedim>>, double> >
	dirichlet_constraints;

	/**
	 * %Vector with domain related point constraints
	 *
	 * The first element of each pair is the constraint.
	 *
	 * The second entry of each pair defines the instant of time within a time step, at which the time dependent functions of the corresponding constraint (e.g. the constraint inhomogeneities) are evaluated.
	 * In particular, values in the range [0, 1] are admissible for each entry, where e.g. 0 means that the time dependent functions are evaluated at the time corresponding to the
	 * beginning of the time step and 1 means that they are evaluated at the time corresponding to the end of the time step.
	 *
	 * Typically, the constraints for state and process variables are evaluated at the end of the time step, while constraints for Lagrangian multipliers are evaluated at the instant of time corresponding to
	 * the time integration parameter \f$\alpha\f$.
	 */
	std::vector< std::pair<dealii::SmartPointer<const dealii::GalerkinTools::PointConstraint<spacedim, spacedim>>, double> >
	point_constraints_domain;

	/**
	 * %Vector with interface related point constraints
	 *
	 * The first element of each pair is the constraint.
	 *
	 * The second entry of each pair defines the instant of time within a time step, at which the time dependent functions of the corresponding constraint (e.g. the constraint inhomogeneities) are evaluated.
	 * In particular, values in the range [0, 1] are admissible for each entry, where e.g. 0 means that the time dependent functions are evaluated at the time corresponding to the
	 * beginning of the time step and 1 means that they are evaluated at the time corresponding to the end of the time step.
	 *
	 * Typically, the constraints for state and process variables are evaluated at the end of the time step, while constraints for Lagrangian multipliers are evaluated at the instant of time corresponding to
	 * the time integration parameter \f$\alpha\f$.
	 */
	std::vector< std::pair<dealii::SmartPointer<const dealii::GalerkinTools::PointConstraint<spacedim-1, spacedim>>, double> >
	point_constraints_interface;

	/**
	 * %Vector with independent scalar related constraints
	 *
	 * The first element of each pair is the constraint.
	 *
	 * The second entry of each pair defines the instant of time within a time step, at which the time dependent functions of the corresponding constraint (e.g. the constraint inhomogeneities) are evaluated.
	 * In particular, values in the range [0, 1] are admissible for each entry, where e.g. 0 means that the time dependent functions are evaluated at the time corresponding to the
	 * beginning of the time step and 1 means that they are evaluated at the time corresponding to the end of the time step.
	 *
	 * Typically, the constraints for state and process variables are evaluated at the end of the time step, while constraints for Lagrangian multipliers are evaluated at the instant of time corresponding to
	 * the time integration parameter \f$\alpha\f$.
	 */
	std::vector< std::pair<dealii::SmartPointer<const dealii::GalerkinTools::PointConstraint<0, spacedim>>, double> >
	point_constraints_C;


public:

	/**
	 * The destructor of Constraints essentially checks before destruction that the
	 * Constraints object is not used by other objects. If this is the case, the program
	 * will be aborted.
	 */
	~Constraints();


	/**
	 * Add a DirichletConstraint to Constraints::dirichlet_constraints
	 *
	 * @param[in]	dirichlet_constraint	DirichletConstraint object
	 *
	 * @param[in]	eval_time				Evaluation time of constraint, see Constraints::dirichlet_constraints_domain_eval_time
	 */
	void
	add_dirichlet_constraint(	const dealii::GalerkinTools::DirichletConstraint<spacedim>& dirichlet_constraint,
								const double eval_time = 1.0);

	/**
	 * Add a domain-related PointConstraint to Constraints::point_constraints_domain
	 *
	 * @param[in]	point_constraint		PointConstraint object
	 *
	 * @param[in]	eval_time				Evaluation time of constraint, see Constraints::point_constraints_domain_eval_time
	 */
	void
	add_point_constraint(	const dealii::GalerkinTools::PointConstraint<spacedim, spacedim>& point_constraint,
							const double eval_time = 1.0);

	/**
	 * Add an interface-related PointConstraint to Constraints::point_constraints_interface
	 *
	 * @param[in]	point_constraint		PointConstraint object
	 *
	 * @param[in]	eval_time				Evaluation time of constraint, see Constraints::point_constraints_interface_eval_time
	 */
	void
	add_point_constraint(	const dealii::GalerkinTools::PointConstraint<spacedim-1, spacedim>& point_constraint,
							const double eval_time = 1.0);

	/**
	 * Add an independent scalar related PointConstraint to Constraints::point_constraints_C
	 *
	 * @param[in]	point_constraint		PointConstraint object
	 *
	 * @param[in]	eval_time				Evaluation time of constraint, see Constraints::point_constraints_C_eval_time
	 */
	void
	add_point_constraint(	const dealii::GalerkinTools::PointConstraint<0, spacedim>& point_constraint,
							const double eval_time = 1.0);

	/**
	 * @return A vector with pointers to the DirichletConstraint objects in Constraints::dirichlet_constraints
	 */
	const std::vector< const dealii::GalerkinTools::DirichletConstraint<spacedim>* >
	get_dirichlet_constraints()
	const;

	/**
	 * @return A vector with pointers to the domain related constraint objects in Constraints::point_constraints_domain
	 */
	const std::vector< const dealii::GalerkinTools::PointConstraint<spacedim, spacedim>* >
	get_point_constraints_domain()
	const;

	/**
	 * @return A vector with pointers to the interface related constraint objects in Constraints::point_constraints_interface
	 */
	const std::vector< const dealii::GalerkinTools::PointConstraint<spacedim-1, spacedim>* >
	get_point_constraints_interface()
	const;

	/**
	 * @return A vector with pointers to the independent scalar related constraint objects in Constraints::point_constraints_C
	 */
	const std::vector< const dealii::GalerkinTools::PointConstraint<0, spacedim>* >
	get_point_constraints_C()
	const;

	/**
	 * @return The independent scalars involved in the definition of the constraints.
	 */
	const std::set< const dealii::GalerkinTools::IndependentField<0, spacedim>* >
	get_independent_scalars()
	const;

	/**
	 * Update the evaluation time of the constraints for the upcoming time-step
	 *
	 * @param[in]	begin_time_step		Time at the start of the time step
	 *
	 * @param[in]	end_time_step		Time at the end of the time step
	 *
	 */
	void
	set_time(	const double begin_time_step,
				const double end_time_step)
	const;

};

}

#endif /* INCREMENTALFE_CONSTRAINTS_H_ */
