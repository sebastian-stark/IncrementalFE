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

#ifndef INCREMENTALFE_GLOBALDATAINCREMENTALFE_H_
#define INCREMENTALFE_GLOBALDATAINCREMENTALFE_H_

#include <incremental_fe/config.h>

#include <string>
#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/base/subscriptor.h>

namespace incrementalFE
{

/**
 * This class is used to store global data like
 * time, reference time, time step number, iteration number, error messages, termination criteria, etc.
 * for the solution of time dependent, non-linear problems.
 *
 * In particular if objects of classes derived from ScalarFunctional
 * or ScalarFunctional<spacedim, spacedim> know of the GlobalDataIncrementalFE
 * object in use, global information (e.g. the current time and the reference time) is available at
 * the material point level. The latter is important for the implementation of time discrete schemes.
 *
 * The GlobalDataIncrementalFE class inherits from Subscriptor in order to be
 * able to check that GlobalDataIncrementalFE objects are only destroyed when they are
 * not needed anymore by other objects.
 *
 * @tparam	spacedim	spatial dimension of the problem
 */
template<unsigned int spacedim>
class GlobalDataIncrementalFE : public dealii::Subscriptor
{

private:

	/**
	 * current time
	 */
	double
	t;

	/**
	 * reference time (previous time)
	 */
	double
	t_ref;

	/**
	 * previous reference time
	 */
	double
	t_ref_old;

	/**
	 * current time step number
	 */
	unsigned int
	time_step = 0;

	/**
	 * If @p true: Use symmetric solver if available in FEModel::solver_wrapper
	 */
	bool
	sym_mode = false;

	/**
	 * Maximum number of iterations allowed for solution of a single time step
	 */
	unsigned int
	max_iter = 20;

	/**
	 * Maximum number of bisections allowed within line search to complete an iteration of a single time step
	 */
	unsigned int
	max_cutbacks = 10;

	/**
	 * Termination criterion for Newton-Raphson iteration.
	 * If sqrt(RHS*DU) (RHS: right hand side of finite element system DU: solution increment) is below this threshold,
	 * the Newton-Raphson iteration of a time increment is terminated.
	 */
	double
	threshold_potential_increment = 1e-14;

	/**
	 * Termination criterion for Newton-Raphson iteration. If the 2-norm of the (scaled) residual is below this threshold,
	 * the Newton-Raphson iteration of a time increment is terminated.
	 * If @p threshold_residual is negative, checking is disabled (default).
	 */
	double
	threshold_residual = -1.0;

	/**
	 * Termination criterion for Newton-Raphson iteration. If the maximum step size is below this threshold,
	 * the Newton-Raphson iteration of a time increment is terminated.
 	 * If @p threshold_residual is negative, checking is disabled (default).
	 */
	double
	threshold_step_size = -1.0;

	/**
	 * If this is set to @p true, only one iteration is performed for each time step.
	 * This parameter should be set to @p true for problems which are known to be linear
	 * (in order to avoid a second iteration),
	 * but must be set to @p false for problems which are nonlinear (the variable is set to @p false by default).
	 */
	bool
	force_linear = false;

	/**
	 * Error messages during solution process.
	 * Objects knowing the GlobalDataIncrementalFE may use the function write_error_message()
	 * to add an error message.
	 */
	std::vector<std::string>
	error_messages;

	/**
	 * Boolean used internally to make sure that time can be reset by GlobalDataIncrementalFE::reset_t()
	 * only once after a call of GlobalDataIncrementalFE::set_t().
	 */
	bool
	reset_t_possible = false;

	/**
	 * Indicate whether a predictor-corrector algorithm is used for time integration. In the predictor-corrector algorithm,
	 * the entire Newton-Raphson iteration scheme is performed twice within each time step (one time with GlobalDataIncrementalFE::predictor_step set to @p true,
	 * and one time with GlobalDataIncrementalFE::predictor_step set to @p false). The starting values for the second Newton-Raphson round are the results of the first round.
	 * The user can use the value of GlobalDataIncrementalFE::predictor_step within the scalar functional definitions to distinguish between predictor and corrector step; and hidden
	 * variables can be used to store local information from the predictor step to be available during the corrector step (e.g., it is possible to store the resulting local dependent variables
	 * at the end of each predictor step in the hidden variables and then base the corrector on these values).
	 */
	bool
	predictor_corrector = false;

	/**
	 * Indicate whether this is a predictor step for time integration schemes of predictor-corrector type.
	 */
	bool
	predictor_step = false;

	/**
	 * Safety distance to an inadmissible state within a single Newton-Raphson iteration (0 < GlobalDataIncrementalFE::safety_distance < 1.0).
	 * The Newton step length will be decreased such that the "distance" between the solution and the
	 * boundary of the domain of admissibility is decreased by GlobalDataIncrementalFE::safety_distance at most during a single iteration (1.0 would correspond to no safety distance at all).
	 * This is used to avoid ill-conditioning problems resulting from a too quick approach of the
	 * boundary of the domain of admissibility.
	 */
	double
	safety_distance = 0.9;

	/**
	 * Level of output written to stdout. Currently only 0 (print no output) and 1 (print output) are possible.
	 */
	unsigned int
	output_level = 1;

	/**
	 * If true, line search is performed during Newton-Raphson iteration
	 */
	bool
	perform_line_search = true;

	/**
	 * Key indicating when to rebuild the sparsity pattern:<br>
	 * 0 - in the beginning of each FEModel::do_time_step() (default)<br>
	 * 1 - only before the next time step (afterwards, GlobalDataIncrementalFE::analyze will be set to 2)<br>
	 * >=2 - do not rebuild
	 *
	 * @warning		The user must ensure that  GlobalDataIncrementalFE::compute_sparsity_pattern=0 or  GlobalDataIncrementalFE::compute_sparsity_pattern=1
	 * 				whenever the sparsity pattern of the matrix needs to be adjusted (during the first time step; when constraints have changed; when dofs have changed; ...).
	 * 				Currently, no internal checking is performed.
	 */
	unsigned int
	compute_sparsity_pattern = 0;

	/**
	 * If this is set to @p false, the Newton-Raphson iteration is not terminated irrespective of the residual.
	 */
	bool
	converged_at_local_level = true;

	/**
	 * If this is set to @p true, the previous solution increment is used to obtain an initial guess for the Newton-Raphson iteration
	 */
	bool
	use_previous_increment_for_initial_guess = false;

	/**
	 * determines whether to use rhs scaling for residual calculation
	 */
	bool
	scale_residual = true;

	/**
	 * if true, the solution is not reset to the solution in the beginning of the time-increment in case the maximum number of iterations is exceeded
	 */
	bool
	continue_on_nonconvergence = false;

	/**
	 * Allow the FEModel class to directly access all members.
	 */
	template <unsigned int, class SolutionVectorType, class RHSVectorType, class MatrixType> friend class FEModel;

public:

	/**
	 * Constructor allowing to construct object with non-default initial time
	 *
	 * @param[in]	t_init	initial time
	 */
	GlobalDataIncrementalFE(const double t_init = 0.0);

	/**
	 * The destructor of GlobalDataIncrementalFE essentially checks before destruction that the
	 * GlobalDataIncrementalFE object is not used by other objects. If this is the case, the program
	 * will be aborted.
	 */
	~GlobalDataIncrementalFE();

	/**
	 * @return	The current time GlobalDataIncrementalFE::t
	 */
	double
	get_t()
	const;

	/**
	 * @return	The reference time GlobalDataIncrementalFE::t_ref
	 */
	double
	get_t_ref()
	const;

	/**
	 * @return	current time step number GlobalDataIncrementalFE::time_step
	 */
	unsigned int
	get_time_step()
	const;

	/**
	 * %Function to write an error message to GlobalDataIncrementalFE::error_messages. This can be used
	 * in scalar functionals to provide with debug information.
	 *
	 * @param[in]	error_message	string with the error message
	 */
	void
	write_error_message(const std::string error_message);

	/**
	 * %Function to write an error message to GlobalDataIncrementalFE::error_messages
	 * recording the spatial location at which the error happened (if the error can
	 * be related to a point in space). This can be used
	 * in scalar functionals to provide with debug information.
	 *
	 * @param[in]	error_message	string with the error message
	 *
	 * @param[in]	location		spatial location at which error occurred
	 */
	void
	write_error_message(const std::string 				error_message,
						const dealii::Point<spacedim>& 	location);

	/**
	 * %Function to print error messages to screen
	 */
	void
	print_error_messages()
	const;

	/**
	 * %Function to print last error message to screen
	 */
	void
	print_last_error_message()
	const;

	/**
	 * Sets GlobalDataIncrementalFE::sym_mode
	 *
	 * @param[in]	sym_mode		GlobalDataIncrementalFE::sym_mode
	 */
	void
	set_sym_mode(const bool sym_mode);

	/**
	 * Sets GlobalDataIncrementalFE::max_iter
	 *
	 * @param[in]	max_iter	GlobalDataIncrementalFE::max_iter
	 *
	 */
	void
	set_max_iter(const unsigned int max_iter);

	/**
	 * Sets GlobalDataIncrementalFE::max_cutbacks
	 *
	 * @param[in]	max_cutbacks	GlobalDataIncrementalFE::max_cutbacks
	 *
	 */
	void
	set_max_cutbacks(const unsigned int max_cutbacks);

	/**
	 * Sets GlobalDataIncrementalFE::threshold_potential_increment
	 *
	 * @param[in]	threshold_potential_increment	GlobalDataIncrementalFE::threshold_potential_increment;
	 */
	void
	set_threshold_potential_increment(const double threshold_potential_increment);

	/**
	 * Sets GlobalDataIncrementalFE::threshold_residual
	 *
	 * @param[in]	threshold_residual	GlobalDataIncrementalFE::threshold_potential_residual;
	 */
	void
	set_threshold_residual(const double threshold_residual);

	/**
	 * Sets GlobalDataIncrementalFE::threshold_step_size
	 *
	 * @param[in]	threshold_step_size	GlobalDataIncrementalFE::threshold_step_size;
	 */
	void
	set_threshold_step_size(const double threshold_step_size);

	/**
	 * Sets GlobalDataIncrementalFE::force_linear
	 *
	 * @param[in]	force_linear	GlobalDataIncrementalFE::force_linear
	 *
	 */
	void
	set_force_linear(const bool force_linear=true);

	/**
	 * Sets GlobalDataIncrementalFE::predictor_corrector
	 *
	 * @param[in]	predictor_corrector		Indicate whether predictor-corrector algorithm is to be used
	 */
	void
	set_predictor_corrector(const bool predictor_corrector = true);

	/**
	 * @return						GlobalDataIncrementalFE::predictor_step
	 */
	bool
	get_predictor_step()
	const;

	/**
	 * @return	GlobalDataIncrementalFE::output_level
	 */
	unsigned int
	get_output_level()
	const;

	/**
	 * Sets GlobalDataIncrementalFE::output_level
	 *
	 * @param[in]	output_level	GlobalDataIncrementalFE::output_level
	 */
	void
	set_output_level(const unsigned int output_level);

	/**
	 * Sets GlobalDataIncrementalFE::safety_distance
	 *
	 * @param[in]	safety_distance	GlobalDataIncrementalFE::safety_distance
	 */
	void
	set_safety_distance(const double safety_distance);

	/**
	 * %Function setting current time (old time is then stored in GlobalDataIncrementalFE::t_ref).
	 * This function also increments the time step number GlobalDataIncrementalFE::time_step
	 * and checks that the new time is larger than the old time.
	 *
	 * @param[in]	t	current time
	 */
	void
	set_t(const double t);

	/**
	 * %Function resetting time to previous step. This undoes the last call to
	 * GlobalDataIncrementalFE::set_t(). The function can only be called once
	 * after a call of GlobalDataIncrementalFE::set_t().
	 */
	void
	reset_t();

	/**
	 * Sets GlobalDataIncrementalFE::predictor_step
	 *
	 * @param[in]	predictor_step	GlobalDataIncrementalFE::predictor_step
	 */
	void
	set_predictor_step(const bool predictor_step = true);

	/**
	 * Sets GlobalDataIncrementalFE::perform_line_search
	 *
	 * @param[in]	perform_line_search	GlobalDataIncrementalFE::perform_line_search
	 */
	void
	set_perform_line_search(const bool perform_line_search = true);

	/**
	 * reset GlobalDataIncrementalFE::t, GlobalDataIncrementalFE::t_ref, GlobalDataIncrementalFE::t_ref_old,
	 * GlobalDataIncrementalFE::time_step, GlobalDataIncrementalFE::error_messages
	 *
	 * @param[in]	t_init	time to be used for resetting GlobalDataIncrementalFE::t,
	 * 						 GlobalDataIncrementalFE::t_ref, GlobalDataIncrementalFE::t_ref_old
	 */
	void
	reinit(const double t_init = 0.0);

	/**
	 * Sets GlobalDataIncrementalFE::compute_sparsity_pattern
	 *
	 * @param[in]	compute_sparsity_pattern	GlobalDataIncrementalFE::compute_sparsity_pattern
	 */
	void
	set_compute_sparsity_pattern(const unsigned int compute_sparsity_pattern = 0);

	/**
	 * Sets GlobalDataIncrementalFE::converged_at_local_level to @p false
	 */
	void
	set_not_converged_at_local_level();

	/**
	 * Sets GlobalDataIncrementalFE::use_previous_increment_for_initial_guess
	 *
	 * @param[in]	use_previous_increment_for_initial_guess	GlobalDataIncrementalFE::use_previous_increment_for_initial_guess
	 */
	void
	set_use_previous_increment_for_initial_guess(const bool use_previous_increment_for_initial_guess = true);

	/**
	 * Sets GlobalDataIncrementalFE::scale_residual
	 *
	 * @param[in]	scale_residual	GlobalDataIncrementalFE::scale_residual
	 */
	void
	set_scale_residual(const bool scale_residual = true);

	/**
	 * Sets GlobalDataIncrementalFE::continue_on_nonconvergence
	 *
	 * @param[in]	continue_on_nonconvergence	GlobalDataIncrementalFE::continue_on_nonconvergence
	 *
	 */
	void
	set_continue_on_nonconvergence(const bool continue_on_nonconvergence=true);

};

}

#endif /* INCREMENTALFE_GLOBALDATAINCREMENTALFE_H_ */
