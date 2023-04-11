// --------------------------------------------------------------------------
// Copyright (C) 2023 by Sebastian Stark
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

#include <deal.II/lac/vector.h>
#include <incremental_fe/manufactured_solution.h>

using namespace std;
using namespace dealii;
using namespace incrementalFE;

#ifdef INCREMENTAL_FE_WITH_SPLINE

template<class SolutionVectorType>
ManufacturedSolutionSpline<SolutionVectorType>::ManufacturedSolutionSpline(	const vector<double>& 				times,
																			const vector<std::vector<double>>&	solution)
{
	splines.reserve(solution.size());
	for(unsigned int m = 0; m < solution.size(); ++m)
		splines.push_back(tk::spline(times, solution[m]));
}



template<class SolutionVectorType>
void
ManufacturedSolutionSpline<SolutionVectorType>::get_manufactured_solution(	const double 		t,
																			SolutionVectorType& solution,
																			const unsigned int	der)
const
{
	solution.reinit(splines.size());
	if(der == 0)
	{
		for(unsigned int m = 0; m < solution.size(); ++m)
			solution[m] = splines[m](t);
	}
	else
	{
		for(unsigned int m = 0; m < solution.size(); ++m)
			solution[m] = splines[m].deriv(der, t);
	}
}

#endif // INCREMENTAL_FE_WITH_SPLINE

template class incrementalFE::ManufacturedSolution<Vector<double>>;
#ifdef INCREMENTAL_FE_WITH_SPLINE
template class incrementalFE::ManufacturedSolutionSpline<Vector<double>>;
#endif // INCREMENTAL_FE_WITH_SPLINE
