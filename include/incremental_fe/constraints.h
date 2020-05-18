#ifndef INCREMENTALFE_CONSTRAINTS_H_
#define INCREMENTALFE_CONSTRAINTS_H_

#include <vector>
#include <atomic>

#include <deal.II/base/subscriptor.h>

#include <galerkin_tools/dirichlet_constraint.h>

namespace incrementalFE
{

/**
 * Class collecting the constraints of the problem. Currently, the class
 * doesn't do much and just collects DirichletConstraint objects of the
 * GalerkinTools library. For this purpose, a @p std::vector could be used as well.
 * However, in the future, the GalerkinTools library may
 * allow for more sophisticated constraints; and then this approach may no
 * longer be feasible.
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
	 */
	std::vector< dealii::SmartPointer<const dealii::GalerkinTools::DirichletConstraint<spacedim>> >
	dirichlet_constraints;

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
	 */
	void
	add_dirichlet_constraint(const dealii::GalerkinTools::DirichletConstraint<spacedim>& dirichlet_constraint);


	/**
	 * @return A vector with pointers to the DirichletConstraint objects in Constraints::dirichlet_constraints
	 */
	const std::vector< const dealii::GalerkinTools::DirichletConstraint<spacedim>* >
	get_dirichlet_constraints()
	const;

	/**
	 * @return The independent scalars involved in the definition of the Dirichlet constraints.
	 */
	const std::set< const dealii::GalerkinTools::IndependentField<0, spacedim>* >
	get_independent_scalars()
	const;

};

}

#endif /* INCREMENTALFE_CONSTRAINTS_H_ */
