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

