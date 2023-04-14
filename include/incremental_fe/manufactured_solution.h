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

#ifndef INCREMENTALFE_MANUFACTUREDSOLUTION_H_
#define INCREMENTALFE_MANUFACTUREDSOLUTION_H_

#include <vector>
#include <incremental_fe/config.h>

#ifdef INCREMENTAL_FE_WITH_SPLINE
#include <spline.h>
#endif // INCREMENTAL_FE_WITH_SPLINE


namespace incrementalFE
{

/**
 * Class providing with a manufactured solution for convergence testing
 *
 * @todo This does not work in parallel yet
 *
 * @tparam SolutionVectorType	Type of the solution vector used in FEModel
 */
template<class SolutionVectorType>
class ManufacturedSolution
{
public:

	/**
	 * Function returning the manufactured solution
	 *
	 * @param[in]	t			Instant of time for evaluation of manufactured solution
	 *
	 * @param[out]	solution	DOF values of manufactured solution evaluated at time t
	 *
	 * @param[in]	der			if greater than 0, store the @p der derivative in @p solution
	 */
	virtual
	void
	get_manufactured_solution(	const double 		t,
								SolutionVectorType& solution,
								const unsigned int	der = 0)
	const = 0;

	/**
	 * virtual destructor
	 */
	virtual
	~ManufacturedSolution() = default;
};

#ifdef INCREMENTAL_FE_WITH_SPLINE
/**
 * This class constructs a manufactured solution based on spline interpolation of a discrete dof solution at certain time instants.
 *
 * This functionality is based on the spline library of T. Kluge (https://kluge.in-chemnitz.de/opensource/spline/).
 * If you want to use it, you must pass the path to the header file to cmake by a flag -DSPLINE_DIR=/path/to/spline/header.
 *
 * @tparam SolutionVectorType	Type of the solution vector used in FEModel
 */
template<class SolutionVectorType>
class ManufacturedSolutionSpline : public ManufacturedSolution<SolutionVectorType>
{
	/**
	 * The vector with the spline representation of the manufactured solution. Each entry corresponds to a single dof.
	 */
	std::vector<tk::spline>
	splines;

public:

	/**
	 * Constructor.
	 *
	 * @param[in]	times		The instants of time at which discrete dof solutions are passed with the second argument
	 *
	 * @param[in]	solution	The discrete dof solution upon which spline interpolation is based. solution[m][n] is the value of dof [m] at time times[n].
	 */
	ManufacturedSolutionSpline(	const std::vector<double>& 				times,
								const std::vector<std::vector<double>>&	solution);


	/**
	 * Function returning the manufactured solution
	 *
	 * @param[in]	t			Instant of time for evaluation of manufactured solution
	 *
	 * @param[out]	solution	DOF values of manufactured solution evaluated at time t
	 *
	 * @param[in]	der			if greater than 0, store the @p der derivative in @p solution
	 */
	void
	get_manufactured_solution(	const double 		t,
								SolutionVectorType& solution,
								const unsigned int	der = 0)
	const
	override
	final;
};

#endif // INCREMENTAL_FE_WITH_SPLINE

}

#endif /* INCREMENTALFE_MANUFACTUREDSOLUTION_H_ */
