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
Constraints<spacedim>::add_dirichlet_constraint(const DirichletConstraint<spacedim>& 	dirichlet_constraint,
												const double 							eval_time)
{
	dirichlet_constraints.push_back(make_pair(&dirichlet_constraint, eval_time));
}

template<unsigned int spacedim>
void
Constraints<spacedim>::add_point_constraint(const PointConstraint<spacedim, spacedim>& 	point_constraint,
											const double 								eval_time)
{
	point_constraints_domain.push_back(make_pair(&point_constraint, eval_time));
}

template<unsigned int spacedim>
void
Constraints<spacedim>::add_point_constraint(const PointConstraint<spacedim-1, spacedim>& 	point_constraint,
											const double 									eval_time)
{
	point_constraints_interface.push_back(make_pair(&point_constraint, eval_time));
}

template<unsigned int spacedim>
void
Constraints<spacedim>::add_point_constraint(const PointConstraint<0, spacedim>& 	point_constraint,
											const double 							eval_time)
{
	point_constraints_C.push_back(make_pair(&point_constraint, eval_time));
}

template<unsigned int spacedim>
const vector< const DirichletConstraint<spacedim>* >
Constraints<spacedim>::get_dirichlet_constraints()
const
{
	vector< const DirichletConstraint<spacedim>* > return_;
	for(const auto& constraint : dirichlet_constraints)
		return_.push_back(constraint.first);
	return return_;
}

template<unsigned int spacedim>
const vector< const PointConstraint<spacedim, spacedim>* >
Constraints<spacedim>::get_point_constraints_domain()
const
{
	vector< const PointConstraint<spacedim, spacedim>* > return_;
	for(const auto& constraint : point_constraints_domain)
		return_.push_back(constraint.first);
	return return_;
}

template<unsigned int spacedim>
const vector< const PointConstraint<spacedim-1, spacedim>* >
Constraints<spacedim>::get_point_constraints_interface()
const
{
	vector< const PointConstraint<spacedim-1, spacedim>* > return_;
	for(const auto& constraint : point_constraints_interface)
		return_.push_back(constraint.first);
	return return_;
}

template<unsigned int spacedim>
const vector< const PointConstraint<0, spacedim>* >
Constraints<spacedim>::get_point_constraints_C()
const
{
	vector< const PointConstraint<0, spacedim>* > return_;
	for(const auto& constraint : point_constraints_C)
		return_.push_back(constraint.first);
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
		if(constraint.first->independent_scalar != nullptr)
			ret.insert(constraint.first->independent_scalar);
	}
	for(const auto& constraint : point_constraints_C)
	{
		ret.insert(constraint.first->independent_field);
	}

	return ret;
}

template<unsigned int spacedim>
void
Constraints<spacedim>::set_time(const double begin_time_step,
								const double end_time_step)
const
{
	for(const auto& constraint : dirichlet_constraints)
	{
		const double t = begin_time_step + constraint.second * (end_time_step - begin_time_step);
		constraint.first->set_time(t);
	}

	for(const auto& constraint : point_constraints_domain)
	{
		const double t = begin_time_step + constraint.second * (end_time_step - begin_time_step);
		constraint.first->set_time(t);
	}

	for(const auto& constraint : point_constraints_interface)
	{
		const double t = begin_time_step + constraint.second * (end_time_step - begin_time_step);
		constraint.first->set_time(t);
	}

	for(const auto& constraint : point_constraints_C)
	{
		const double t = begin_time_step + constraint.second * (end_time_step - begin_time_step);
		constraint.first->set_time(t);
	}

}

template class incrementalFE::Constraints<2>;
template class incrementalFE::Constraints<3>;

