#ifndef INCREMENTALFE_GLOBALDATAINCREMENTALFE_H_
#define INCREMENTALFE_GLOBALDATAINCREMENTALFE_H_

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
 * the material point level. The latter is important for the implementation of incremental
 * potentials.
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
	 * If @p true: Use symmetric solver
	 */
	bool
	sym_mode = false;

	/**
	 * Maximum number of iterations allowed for solution of a single time step
	 */
	unsigned int
	max_iter = 20;

	/**
	 * Maximum number of cutbacks allowed within line search to complete an iteration of a single time step
	 */
	unsigned int
	max_cutbacks = 10;

	/**
	 * Termination criterion for Newton-Raphson iteration.
	 * If the estimated increment in the (quadratically approximated) potential is below this threshold,
	 * the Newton-Raphson iteration of a single time step is terminated.
	 */
	double
	threshold_potential_increment = 1e-14;

	/**
	 * If this is set to @p true, only one iteration is performed for each time step.
	 * This parameter should be set to @p true for problems which are known to be linear
	 * (in order to avoid a second iteration),
	 * but must be set to @p false for problems which are nonlinear.
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
	 * Indicate whether a predictor-corrector algorithm is used for time integration
	 */
	bool
	predictor_corrector = false;

	/**
	 * Indicate whether this is a predictor step for time integration schemes of predictor-corrector type.
	 */
	bool
	predictor_step = false;

	/**
	 * Safety distance to an inadmissible state within a single Newton-Raphson iteration (<1).
	 * The Newton step length will be decreased such that the "distance" between the solution and the
	 * boundary of the domain of admissibility is decreased by 90% at most during a single iteration.
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
	 * @return	current time step number GlobalDataIncrementalFE::time_step
	 */
	unsigned int
	get_time_step()
	const;

	/**
	 * %Function to write an error message to GlobalDataIncrementalFE::error_messages
	 *
	 * @param[in]	error_message	string with the error message
	 */
	void
	write_error_message(const std::string error_message);

	/**
	 * %Function to write an error message to GlobalDataIncrementalFE::error_messages
	 * recording the spatial location at which the error happened (if the error can
	 * be related to a point in space)
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
	 * Sets GlobalDataIncrementalFE::predictor_step
	 *
	 * @param[in]					GlobalDataIncrementalFE::predictor_step
	 */
	void
	set_predictor_step(const bool predictor_step = true);

	/**
	 * @return						GlobalDataIncrementalFE::output_level
	 */
	unsigned int
	get_output_level()
	const;

	/**
	 * Sets GlobalDataIncrementalFE::output_level
	 *
	 * @param[in]					GlobalDataIncrementalFE::output_level
	 */
	void
	set_output_level(const unsigned int output_level);

	/**
	 * reset GlobalDataIncrementalFE::t, GlobalDataIncrementalFE::t_ref, GlobalDataIncrementalFE::t_ref_old,
	 * GlobalDataIncrementalFE::time_step, GlobalDataIncrementalFE::error_messages
	 *
	 * @param[in]	t_init	time to be used for resetting GlobalDataIncrementalFE::t,
	 * 						 GlobalDataIncrementalFE::t_ref, GlobalDataIncrementalFE::t_ref_old
	 */
	void
	reinit(const double t_init = 0.0);

};

}

#endif /* INCREMENTALFE_GLOBALDATAINCREMENTALFE_H_ */
