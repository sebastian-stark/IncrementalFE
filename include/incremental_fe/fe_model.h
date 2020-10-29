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

#ifndef INCREMENTALFE_FEMODEL_H_
#define INCREMENTALFE_FEMODEL_H_

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/smartpointer.h>
#include <galerkin_tools/assembly_helper.h>
#include <galerkin_tools/solver_wrapper.h>
#include <galerkin_tools/two_block_sparsity_pattern.h>
#include <galerkin_tools/dof_renumbering.h>
#include <incremental_fe/constraints.h>
#include <incremental_fe/global_data_incremental_fe.h>
#include <boost/signals2.hpp>

namespace incrementalFE
{

/**
 * Class for solution of non-linear, transient problems by one-step time integration.
 *
 * The solution algorithm within a single time step is based on a Newton-Raphson
 * scheme together with a residual based line search algorithm enhancing convergence.
 *
 * @tparam	SolutionVectorType	the type used for the solution vector and the rhs,
 * 								must be consistent with the SolverWrapper used
 * 								(in parallel this vector type must permit read access to ghosted entries while write access is not required)
 *
 * @tparam	RHSVectorType		the type used for the rhs,
 * 								must be consistent with the SolverWrapper used
 * 								(in parallel this vector type must permit write access to ghosted entries while read access is not required)

 * @tparam	MatrixType			the type used for the system matrix,
 * 								must be consistent with the SolverWrapper used
 *
 */
template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
class FEModel
{

private:

	/**
	 * Object defining dof renumbering scheme of assembly helper (only relevant in parallel)
	 */
	dealii::GalerkinTools::DoFRenumberingOffset
	dof_renumbering;

	/**
	 * The AssemblyHelper object defining the problem to be solved
	 */
	dealii::GalerkinTools::AssemblyHelper<spacedim>
	assembly_helper;

	/**
	 * The GlobalDataIncrementalFE object, which is used to exchange global information about
	 * the finite element model, the solution process, etc.
	 * In particular, this object contains the time step information, which
	 * may be needed within the TotalPotential to define the space-time discrete formulation.
	 */
	const dealii::SmartPointer<GlobalDataIncrementalFE<spacedim>>
	global_data;

	/**
	 * The Constraints object comprising all the constraints to be applied
	 */
	const dealii::SmartPointer<const Constraints<spacedim>>
	constraints;

	/**
	 * The SolverWrapper provides the functionality to solve the linear systems within each Newton-Raphson iteration
	 */
	const dealii::SmartPointer<dealii::GalerkinTools::SolverWrapper<SolutionVectorType, RHSVectorType, MatrixType, dealii::GalerkinTools::TwoBlockSparsityPattern>>
	solver_wrapper;

	/**
	 * The sparsity pattern of the system matrix FEModel::system_matrix (used internally during the solution process)
	 */
	dealii::GalerkinTools::TwoBlockSparsityPattern
	sparsity_pattern;

	/**
	 * System matrix (used internally during the solution process)
	 */
	MatrixType
	system_matrix;

	/**
	 * Solution vector with the current solution
	 */
	SolutionVectorType
	solution;

	/**
	 * Solution vector with the reference solution
	 */
	SolutionVectorType
	solution_ref;

	/**
	 * Solution vector with the solution increment during the last successfully completed time increment. This is used to
	 * obtain an initial guess for the Newton-Raphson iteration in case GlobalDataIncrementalFE::use_previous_increment_for_initial_guess == @p true
	 */
	SolutionVectorType
	delta_solution_last_step;

	/**
	 * Right hand side vector (used internally during the solution process)
	 */
	RHSVectorType
	rhs;

	/**
	 * Current value of the total potential (used internally during the solution process)
	 */
	double
	potential_value = 0.0;

	/**
	 * %Vector used to scale elements of FEModel::rhs before computing 2-norm thereof (each element of FEModel::rhs is scaled by the reciprocal value of the corresponding element of  FEModel::rhs_scaling_vector).
	 * This scaling is used to avoid ill-conditioning problems.
	 * The FEModel::rhs_scaling_vector is computed in the beginning of each time step based on the initial system matrix and then kept constant
	 * throughout the time step. Each element in FEModel::rhs_scaling_vector is equal to the maximum norm of the corresponding row in the initial system matrix.
	 */
	RHSVectorType
	rhs_scaling_vector;

	/**
	 * Matrix with hanging node constraints (used internally during the solution process)
	 */
	dealii::AffineConstraints<double>
	hanging_node_constraints;

	/**
	 * Matrix with all combined constraints (used internally during the solution process)
	 */
	dealii::AffineConstraints<double>
	all_constraints;

	/**
	 * Pairs of time and filename of outputs written to files (for domain output).
	 * This is used in order to generate files gathering all the output files of the individual time steps into one
	 * file, which can be opened for post-processing.
	 */
	std::vector< std::pair<double, std::string> >
	times_and_names_domain;

	/**
	 * Pairs of time and filename of outputs written to files (for interface output).
	 * This is used in order to generate files gathering all the output files of the individual time steps into one
	 * file, which can be opened for post-processing.
	 */
	std::vector< std::pair<double, std::string> >
	times_and_names_interface;

	/**
	 * A list of connections set up by FEModel. This allows for disconnection when the FEModel object is destructed
	 */
	std::vector<boost::signals2::connection>
	connections;

	/**
	 * DataPostprocessor objects for domain cells. This allows for user-defined post-processing.
	 */
	std::vector<dealii::SmartPointer<const dealii::DataPostprocessor<spacedim>>>
	dp_domain;

	/**
	 * DataPostprocessor objects for interface cells. This allows for user-defined post-processing.
	 */
	std::vector<dealii::SmartPointer<const dealii::DataPostprocessor<spacedim>>>
	dp_interface;

	/**
	 * determines whether hanging node constraints are taken into account or ignored
	 */
	const bool
	make_hanging_node_constraints = true;

	/**
	 * if @p true, system matrix is treated as a single block. This is useful in case the A block of the
	 * block system is singular.
	 */
	const bool
	single_block = false;

	/**
	 * total time spent for linear solver during last time step
	 */
	double
	solve_time_last_step = 0.0;

	/**
	 * Reinit a solution type vector
	 *
	 * @param[in]	vector	The vector to be re-inited
	 */
	void
	reinit_solution_vector(SolutionVectorType& vector);

	/**
	 * Reinit a rhs type vector
	 *
	 * @param[in]	vector	The vector to be re-inited
	 */
	void
	reinit_rhs_vector(RHSVectorType& vector);

	/**
	 * Reinit a system matrix
	 *
	 * @param[in]	matrix	The system matrix to be re-inited
	 */
	void
	reinit_matrix(MatrixType& matrix);

	/**
	 * %Function computing current FEModel::rhs and FEModel::system_matrix
	 *
	 * @param[in]	solution_ref	The reference solution vector
	 *
	 * @param[in]	constraints		The constraints object
	 *
	 * @return						@p true: error, @p false: no error
	 */
	bool
	compute_system(	const SolutionVectorType& 			solution_ref,
					dealii::AffineConstraints<double>&	constraints);

	/**
	 * %Function computing the FEModel::sparsity_pattern
	 *
	 * @param[in]	constraints		Constraints to be taken into account
	 */
	void
	compute_sparsity_pattern(const dealii::AffineConstraints<double>& constraints);

	/**
	 *
	 * %Function assembling constraints (combines Dirichlet type constraints, hanging node constraints, custom constraints while taking into account
	 * the constraints to be ignored)
	 *
	 * Updates also FEModel::dirichlet_constraints
	 *
	 * @param[out]	constraints			The resulting constraints object
	 *
	 * @param[in]	custom_constraints	Constraints additional to FEModel::constraints to be taken into consideration (should be closed).
	 * 									This allows e.g. for constraints prescribing individual dofs (e.g. to fix a pressure variable)
	 *
	 * @param[in]	ignore_constraints	Constraints object with constraint lines to be ignored. This allows e.g. to eliminate overconstraints
	 * 									at individual vertices resulting e.g. from different constraints imposed on different portions of the boundary.
	 */
	void
	make_constraints(	dealii::AffineConstraints<double>&			constraints,
						const dealii::AffineConstraints<double>&	custom_constraints = dealii::AffineConstraints<double>(),
						const dealii::AffineConstraints<double>&	ignore_constraints = dealii::AffineConstraints<double>());

	/**
	 * %Function adjusting constraint inhomogeneities associated with Dirichlet type constraints such that constraint object
	 * applies to solution increment and not to the solution itself
	 */
	void
	adjust_constraint_inhomogeneity(dealii::AffineConstraints<double>&	constraints)
	const;

	/**
	 * %Function updating FEModel::rhs_scaling_vector according to current FEModel::system_matrix
	 */
	void
	update_rhs_scaling_vector();

	/**
	 * computes estimated potential increment (inner product between @p delta_solution and FEModel::rhs)
	 *
	 * @param[in]	delta_solution		solution increment
	 *
	 * @return							estimated potential increment
	 */
	double
	compute_estimated_potential_increment(const SolutionVectorType& delta_solution)
	const;

	/**
	 * Computes maximum allowable step size without obtaining inadmissible state and adjust @p delta_solution accordingly.
	 * The "safety distance" to the domain of inadmissible states is specified by GlobalDataIncrementalFE::safety_distance.
	 *
	 * @param[inout]	delta_solution	increment direction
	 *
	 * @param[in]		solution_ref	reference solution
	 *
	 * @param[in]		constraints		constraint matrix
	 *
	 * @return							maximum allowable step size
	 */
	void
	adjust_delta_solution(	SolutionVectorType& 						delta_solution,
							const SolutionVectorType& 					solution_ref,
							const dealii::AffineConstraints<double>&	constraints);

	/**
	 * Update the ghost values of @p vector
	 *
	 * @param[in]	vector	The vector for which the ghost values are to be imported
	 */
	void
	update_ghosts(SolutionVectorType& vector);

	/**
	 * Zero the ghost values of @p vector
	 *
	 * @param[in]	vector	The vector for which the ghost values are to be zeroed
	 */
	void
	zero_ghosts(SolutionVectorType& vector);

	/**
	 * @return	The 2-norm of the scaled version of FEModel::rhs, for details regarding the scaling see FEModel::rhs_scaling_vector
	 */
	double
	get_residual()
	const;

	/**
	 * %Function called automatically after the triangulation system is changed
	 */
	void
	post_refinement();

public:

	/**
	 * Constructor
	 *
	 * @param[in]	total_potential					TotalPotential object defining the total potential
	 *
	 * @param[in]	tria_system						TriangulationSystem object
	 *
	 * @param[in]	mapping_domain					Mapping to be used on the domain
	 *
	 * @param[in]	mapping_interface				Mapping to be used on the interfaces
	 *
	 * @param[in]	global_data						The global data object to be used, see FEModel::global_data
	 *
	 * @param[in]	constraints						Constraints object, see FEModel::constraints
	 *
	 * @param[in]	solver_wrapper					SolverWrapper to be used, see FEModel::solver_wrapper
	 *
	 * @param[in]	make_hanging_node_constraints	FEModel::make_hanging_node_constraints
	 *
	 * @param[in]	single_block					FEModel::single_block
	 */
	FEModel(const dealii::GalerkinTools::TotalPotential<spacedim>&																						total_potential,
			dealii::GalerkinTools::TriangulationSystem<spacedim>&																						tria_system,
			const dealii::Mapping<spacedim, spacedim>&																									mapping_domain,
			const dealii::Mapping<spacedim-1, spacedim>&																								mapping_interface,
			GlobalDataIncrementalFE<spacedim>&																											global_data,
			const Constraints<spacedim>&																												constraints,
			dealii::GalerkinTools::SolverWrapper<SolutionVectorType, RHSVectorType, MatrixType, dealii::GalerkinTools::TwoBlockSparsityPattern>&		solver_wrapper,
			const bool																																	make_hanging_node_constraints = true,
			const bool																																	single_block = false);

	/**
	 * Destructor
	 */
	~FEModel();

	/**
	 * Method to compute a single time step
	 *
	 * @param[in]	t					new time
	 *
	 * @param[in]	custom_constraints	Constraints additional to FEModel::constraints to be taken into consideration (should be closed).
	 * 									This allows e.g. for constraints prescribing individual dofs (e.g. to fix a pressure variable)
	 *
	 * @param[in]	ignore_constraints	Constraints object with constraint lines to be ignored. This allows e.g. to eliminate overconstraints
	 * 									at individual vertices resulting e.g. from different constraints imposed on different portions of the boundary.
	 *
	 * @return	Number of Newton-Raphson iterations required for time step, returns -1 if no convergence or other error
	 */
	int
	do_time_step(	const double 								t,
					const dealii::AffineConstraints<double>&	custom_constraints = dealii::AffineConstraints<double>(),
					const dealii::AffineConstraints<double>&	ignore_constraints = dealii::AffineConstraints<double>());

	/**
	 * %Function computing the "distance" of the solution vector FEModel::solution of this FEModel
	 * to the solution vector FEModel::solution of another FEModel (the FEModel objects must be the same
	 * apart from the mesh refinement, in particular they must be based on the same coarse mesh).
	 *
	 * Note that the values of the independent scalars are currently not taken into account in this method.
	 *
	 * For further details see AssemblyHelper::compute_distance_to_other_solution().
	 *
	 * @todo Hanging node constraints are currently not taken care of when comparing the solutions. Also not all VectorTools::NormType norms are implemented yet.
	 *
	 * @param[in]	other_incremental_fe		The other FEModel object
	 *
	 * @param[in]	quadrature_domain			Quadrature scheme to be used on the domain for the computation of the norm
	 *
	 * @param[in]	quadrature_interface		Quadrature scheme to be used on the interface for the computation of the norm
	 *
	 * @param[in]	norm_type					Type of the norm (note: currently only VectorTools::NormType::@p L2_norm and VectorTools::NormType::@p Linfty_norm are implemented)
	 *
	 * @param[in]	component_mask_domain		Domain related solution components to be included in the calculation. If the ComponentMask
	 * 											is empty, all components will be included
	 *
	 * @param[in]	component_mask_interface	Domain related solution components to be included in the calculation. If the ComponentMask
	 * 											is empty, all components will be included
	 *
	 * @param[in]	exponent					Exponent of the norm if required. Currently this is unused because no norms with variable exponent are implemented.
	 *
	 * @param[in]	scaling_domain				Scaling factors to be used for errors of individual solution components on domain
	 *
	 * @param[in]	scaling_interface			Scaling factors to be used for errors of individual solution components on interface
	 *
	 * @return									The value of the norm computed on the domain and the interface, respectively
	 */
	std::pair<const double, const double>
	compute_distance_to_other_solution( const FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>&	other_incremental_fe,
										const dealii::Quadrature<spacedim>										quadrature_domain,
										const dealii::Quadrature<spacedim-1>									quadrature_interface,
										const dealii::VectorTools::NormType										norm_type = dealii::VectorTools::NormType::L2_norm,
										const dealii::ComponentMask												component_mask_domain = dealii::ComponentMask(),
										const dealii::ComponentMask												component_mask_interface = dealii::ComponentMask(),
										const double															exponent = 2.0,
										const dealii::Vector<double>											scaling_domain = dealii::Vector<double>(),
										const dealii::Vector<double>											scaling_interface = dealii::Vector<double>())
	const;

	/**
	 * %Function computing the "distance" of the solution vector FEModel::solution
	 * to an exact solution.
	 *
	 * The exact and the numerical solution are subtracted and finally the norm of the resulting difference is computed numerically.
	 * This is done for the domain related and the interface related part separately.
	 *
	 * Note that the values of the independent scalars are currently not taken into account in this method.
	 *
	 * @param[in]	exact_solution_domain		Exact solution on domain (use AssemblyHelper::get_u_omega_global_component_indices() to obtain information
	 * 											about the component indexing; the underlying AssemblyHelper needed for this can be obtained by FEModel::get_assembly_helper())
	 *
	 * @param[in]	exact_solution_interface	Exact solution on interface (use AssemblyHelper::get_u_sigma_global_component_indices() to obtain information
	 * 											about the component indexing; the underlying AssemblyHelper needed for this can be obtained by FEModel::get_assembly_helper())
	 *
	 * @param[in]	quadrature_domain			Quadrature scheme to be used on the domain for the computation of the norm
	 *
	 * @param[in]	quadrature_interface		Quadrature scheme to be used on the interface for the computation of the norm
	 *
	 * @param[in]	norm_type					Type of the norm
	 *
	 * @param[in]	component_mask_domain		Domain related solution components to be included in the calculation. If the ComponentMask
	 * 											is empty, all components will be included
	 *
	 * @param[in]	component_mask_interface	Domain related solution components to be included in the calculation. If the ComponentMask
	 * 											is empty, all components will be included
	 *
	 * @param[in]	exponent					Exponent of the norm if required
	 *
	 * @return									The value of the norm computed on the domain and the interface, respectively
	 */
	std::pair<const double, const double>
	compute_distance_to_exact_solution(	const dealii::Function<spacedim>&		exact_solution_domain,
										const dealii::Function<spacedim>&		exact_solution_interface,
										const dealii::Quadrature<spacedim>		quadrature_domain,
										const dealii::Quadrature<spacedim-1>	quadrature_interface,
										const dealii::VectorTools::NormType		norm_type = dealii::VectorTools::NormType::L2_norm,
										const dealii::ComponentMask				component_mask_domain = dealii::ComponentMask(),
										const dealii::ComponentMask				component_mask_interface = dealii::ComponentMask(),
										const double							exponent = 2.0)
	const;

	/**
	 * Reads the solution vector from a file
	 *
	 * @param[in]	file_name		Name of the file to be read from (including extension)
	 *
	 * @warning		This function does nothing apart from reading the solution vector previously stored with FEModel::write_solution_to_file
	 * 				back into the solution vector. It is the user's responsibility that the FEModel is left in a
	 * 				usable state afterwards. In particular, no checking for the size of the solution vector or
	 * 				for the satisfaction of constraints is done. Also, this function does not read in the values of
	 * 				the hidden variables at the material points. Essentially, the purpose of this function is to
	 * 				make a computed solution available for post-processing without having to repeat the entire solution
	 * 				process. In particular, this function is not meant for restarting an analysis at a certain point.
	 */
	void
	read_solution_from_file(const std::string file_name);

	/**
	 * Writes the solution vector to a file
	 *
	 * @param[in]	file_name		Name of the file to be written to (including extension)
	 */
	void
	write_solution_to_file(const std::string file_name)
	const;


	/**
	 * %Function to write independent field output. The output is written to *.vtu files
	 * (one for domain related output and one for interface related output).
	 * Generally, the time step number is appended to the file names provided to this function. In
	 * addition, *.pvd records are written for the domain related output and the
	 * interface related output (file names according to the input parameters of this function).
	 *
	 * @param[in]	file_name_domain	File name (without extension) for independent field output on domain
	 *
	 * @param[in]	file_name_interface	File name (without extension) for independent field output on interfaces
	 *
	 * @param[in]	n_subdivisions		The number of subdivisions of each cell (to get a better representation in case of curved inner cells, higher order elements, etc.)
	 */
	void
	write_output_independent_fields(	const std::string 	file_name_domain,
										const std::string 	file_name_interface,
										const unsigned int	n_subdivisions = 1);
	/**
	 * @return the FEModel::assembly_helper object
	 */
	const dealii::GalerkinTools::AssemblyHelper<spacedim>&
	get_assembly_helper();

	/**
	 * @return	FEModel::potential_value
	 */
	double
	get_potential_value()
	const;

	/**
	 * %Function attaching DataPostprocessor for domain cells. The DataPostprocessor objects attached
	 * are used in FEModel::write_output_independent_fields() to generate output.
	 *
	 * @param[in]	dp	The DataPostprocessor object
	 */
	void attach_data_postprocessor_domain(const dealii::DataPostprocessor<spacedim>& dp);

	/**
	 * %Function attaching DataPostprocessor for interface cells. The DataPostprocessor objects attached
	 * are used in FEModel::write_output_independent_fields() to generate output.
	 *
	 * @param[in]	dp	The DataPostprocessor object
	 */
	void attach_data_postprocessor_interface(const dealii::DataPostprocessor<spacedim>& dp);

	/**
	 * @return		const reference to solution vector
	 */
	const SolutionVectorType&
	get_solution_vector()
	const;

	/**
	 * @return		reference to solution vector
	 */
	SolutionVectorType&
	get_solution_vector();


	/**
	 * @return		const reference to reference solution vector
	 */
	const SolutionVectorType&
	get_solution_ref_vector()
	const;

	/**
	 * @return		reference to reference solution vector
	 */
	SolutionVectorType&
	get_solution_ref_vector();

	/**
	 * return FEModel::solve_time_last_step
	 */
	double
	get_solve_time_last_step()
	const;


};

}

#endif /* INCREMENTALFE_FEMODEL_H_ */
