#include <incremental_fe/constraints.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
Constraints<spacedim>::~Constraints()
{
	Assert(n_subscriptions() == 0, ExcMessage("You are about to destroy a Constraints object, which is currently in use! Make sure that all Constraints objects live at least as long as the objects using them!"));
}

template<unsigned int spacedim>
void
Constraints<spacedim>::add_dirichlet_constraint(const DirichletConstraint<spacedim>& dirichlet_constraint)
{
	dirichlet_constraints.push_back(&dirichlet_constraint);
}

template<unsigned int spacedim>
const vector< const DirichletConstraint<spacedim>* >
Constraints<spacedim>::get_dirichlet_constraints()
const
{
	vector< const DirichletConstraint<spacedim>* > return_;
	for(const auto& constraint : dirichlet_constraints)
		return_.push_back(constraint);
	return return_;
}

template<unsigned int spacedim>
const set< const IndependentField<0, spacedim>* >
Constraints<spacedim>::get_independent_scalars()
const
{
	set< const IndependentField<0, spacedim>* > ret;
	for(const auto& constraint : dirichlet_constraints)
	{
		if(constraint->independent_scalar != nullptr)
			ret.insert(constraint->independent_scalar);
	}
	return ret;
}

template class Constraints<2>;
template class Constraints<3>;

