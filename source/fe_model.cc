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

#include <iostream>
#include <fstream>

#ifdef DEAL_II_WITH_PETSC
#include <petscmat.h>
#endif

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/parpack_solver.h>
#include <deal.II/base/conditional_ostream.h>

#include <incremental_fe/fe_model.h>
#include <galerkin_tools/tools.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::FEModel(	const TotalPotential<spacedim>&																				total_potential,
																			TriangulationSystem<spacedim>&																				tria_system,
																			const Mapping<spacedim, spacedim>&																			mapping_domain,
																			const Mapping<spacedim-1, spacedim>&																		mapping_interface,
																			GlobalDataIncrementalFE<spacedim>&																			global_data,
																			const Constraints<spacedim>&																				constraints,
																			const SolverWrapper<SolutionVectorType, RHSVectorType, MatrixType, GalerkinTools::TwoBlockSparsityPattern>&	solver_wrapper,
																			const bool																									make_hanging_node_constraints)
:
assembly_helper(total_potential, tria_system, mapping_domain, mapping_interface, constraints.get_independent_scalars()),
global_data(&global_data),
constraints(&constraints),
solver_wrapper(&solver_wrapper),
make_hanging_node_constraints(make_hanging_node_constraints)
{
	// connect post_refinement() to post refinement signal of triangulation system
	connections.push_back(tria_system.post_refinement.connect(0, boost::bind(&FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::post_refinement, this)));

	// apply dof renumbering scheme in parallel in order to let dofs form a contiguous range on each processor
	if(tria_system.get_this_proc_n_procs().second > 1)
	{
#ifdef DEAL_II_WITH_MPI
		Auxiliary::compute_dof_renumbering_contiguous(assembly_helper.get_dof_handler_system(), dof_renumbering);
		dof_renumbering.add_range(assembly_helper.system_size() - assembly_helper.get_n_stretched_rows(), assembly_helper.system_size() - 1, 0);
		assembly_helper.get_dof_handler_system().attach_dof_renumbering(dof_renumbering);
#else
		Assert(false, ExcMessage("Internal error: deal.II not compiled with MPI, but calculation run with ore than one processor"));
#endif
	}

	// compute hanging node constraints (these need only be updated after mesh refinement)
	if(make_hanging_node_constraints)
		assembly_helper.get_dof_handler_system().make_hanging_node_constraints(hanging_node_constraints);
	hanging_node_constraints.close();

	// compute constraints in initial state
	AffineConstraints<double> initial_constraints(assembly_helper.get_locally_relevant_indices());
	make_constraints(initial_constraints);

	// set initial state
	reinit_solution_vector(solution);
	assembly_helper.get_initial_fields_vector(solution, &initial_constraints);
}


template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::~FEModel()
{
	// disconnect signals
	for(auto &connection : connections)
		connection.disconnect();
	connections.clear();
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
int
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::do_time_step(	const double						t,
																				const AffineConstraints<double>&	custom_constraints,
																				const AffineConstraints<double>&	ignore_constraints)
{
	const unsigned int this_proc = assembly_helper.get_triangulation_system().get_this_proc_n_procs().first;
	ConditionalOStream pout(cout, (this_proc == 0) && (global_data->get_output_level() > 0));

	Timer timer;
	timer.start();

	// in a predictor-corrector algorithm each time step starts with the predictor step
	if(global_data->predictor_corrector)
	{
		pout << "Starting predictor" << endl;
		global_data->predictor_step = true;
	}
	else
		global_data->predictor_step = false;

	// error variable - on error: reset to state before time step and return
	bool error = false;

	// old/reference values of solution
	SolutionVectorType solution_ref = solution;


	// make sure that ghosts are imported
	update_ghosts(solution);
	update_ghosts(solution_ref);

	timer.stop();
	pout << "Elapsed CPU time preparation: " << timer.cpu_time() << " seconds." << endl;
	pout << "Elapsed wall time preparation: " << timer.wall_time() << " seconds." << endl;
	timer.reset();


	// set new time
	global_data->set_t(t);

	timer.start();
	// set up constraints
	AffineConstraints<double> constraints(assembly_helper.get_locally_relevant_indices());
	make_constraints(constraints, custom_constraints, ignore_constraints);
	timer.stop();
	pout << "Elapsed CPU time constraints: " << timer.cpu_time() << " seconds." << endl;
	pout << "Elapsed wall time constraints: " << timer.wall_time() << " seconds." << endl;
	timer.reset();

	timer.start();
	// make sparsity pattern
	compute_sparsity_pattern(constraints);
	timer.stop();
	pout << "Elapsed CPU time sparsity: " << timer.cpu_time() << " seconds." << endl;
	pout << "Elapsed wall time sparsity: " << timer.wall_time() << " seconds." << endl;
	timer.reset();

	// ensure consistency of current solution with constraints
	zero_ghosts(solution);
	constraints.distribute(solution);
	update_ghosts(solution);

	// solution increment of Newton-Raphson iteration
	SolutionVectorType delta_solution;
	reinit_solution_vector(delta_solution);

	// residual value of current iteration and of previous iteration (used later to ensure that the residual is strictly decreasing between subsequent iterations)
	double residual = 0.0;
	double residual_old = 0.0;

	// number of iterations performed
	unsigned int iter = 0;

	// bring constraints in appropriate form for solution increment
	adjust_constraint_inhomogeneity(constraints);

	timer.start();
	// assemble the system for the first iteration (before doing so, bring the constraints in the appropriate form)
	if(compute_system(solution_ref, constraints))
	{
		solution = solution_ref;
		global_data->reset_t();
		return -1;
	}
	timer.stop();
	pout << "Elapsed CPU time assembly: " << timer.cpu_time() << " seconds." << endl;
	pout << "Elapsed wall time assembly: " << timer.wall_time() << " seconds." << endl;
	timer.reset();

	// compute rhs scaling vector
	timer.start();
	update_rhs_scaling_vector();
	timer.stop();
	pout << "Elapsed CPU time scaling: " << timer.cpu_time() << " seconds." << endl;
	pout << "Elapsed wall time scaling: " << timer.wall_time() << " seconds." << endl;
	timer.reset();

	// start iteration loop
	for(;;)
	{
		// solve the system
		timer.start();
		solver_wrapper->solve(system_matrix, delta_solution, rhs, global_data->sym_mode);
		timer.stop();
		pout << "Elapsed CPU time solve: " << timer.cpu_time() << " seconds." << endl;
		pout << "Elapsed wall time solve: " << timer.wall_time() << " seconds." << endl;
		timer.reset();

		// incorporate constraints into solution increment
		zero_ghosts(delta_solution);
		constraints.distribute(delta_solution);
		update_ghosts(delta_solution);

		// only perform check of termination criterion starting from the second iteration
		if(iter > 0)
		{
			// compute estimated potential increment
			double estimated_potential_increment = compute_estimated_potential_increment(delta_solution);

			// check termination criterion
			if(fabs(estimated_potential_increment) < global_data->threshold_potential_increment)
			{
				residual = get_residual();
				pout << "Converged with residual: " << residual << " and potential increment " << estimated_potential_increment << endl << endl;

				// if this was the predictor step, continue with the corrector step, otherwise the time step is completed
				if(global_data->predictor_step)
				{
					global_data->predictor_step = false;
					pout << "Starting corrector" << endl;
					iter = 0;
					continue;
				}
				else
					break;
			}
		}

		// compute maximum step size and adjust delta_solution accordingly (this avoids that an inadmissible solution is obtained)
		if(!global_data->force_linear)
			adjust_delta_solution(delta_solution, solution_ref, constraints);


		// update solution
		solution += delta_solution;

		// adjust constraint inhomogeneity according to new solution
		adjust_constraint_inhomogeneity(constraints);

		// if the problem is linear, quit here
		if(global_data->force_linear)
		{
			pout << "Linear step" << endl;
			break;
		}

		// perform line search to ensure that residual is strictly decreasing (in this context, the new system matrix and rhs are computed as well
		unsigned int cutbacks = 0;
		for(;;)
		{
			// compute new rhs and system matrix here (update constraints before)
			if(compute_system(solution_ref, constraints))
			{
				error = true;
				break;
			}

			residual = get_residual();
			if( (iter == 0) || (residual < residual_old) )
			{
				break;
			}
			else
			{
				++cutbacks;
				pout << "CUTBACK" << endl;
				if(cutbacks > global_data->max_cutbacks)
				{
					global_data->write_error_message("Exceeded the allowed number of cutbacks, solution unconverged!");
					error = true;
					break;
				}

				delta_solution *= -0.5;
				zero_ghosts(delta_solution);
				constraints.distribute(delta_solution);
				update_ghosts(delta_solution);

				solution += delta_solution;

				delta_solution *= -1.0;
			}
		}
		if(error == true)
			break;

		residual_old = residual;

		++iter;

		//stop if allowed number of iterations is exceeded
		if(iter > global_data->max_iter)
		{
			global_data->write_error_message("Exceeded the allowed number of iterations, solution unconverged!");
			error = true;
			break;
		}

		pout << "Iteration " << iter << endl;
		pout << "Residual:" << residual << endl;

	}

	//if no success, reset to old state here
	if(error)
	{
		solution = solution_ref;
		global_data->reset_t();
		return -1;
	}
	else
		return iter;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
std::pair<const double, const double>
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::compute_distance_to_other_solution(	const FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>&	other_incremental_fe,
																										const Quadrature<spacedim>												quadrature_domain,
																										const Quadrature<spacedim-1>											quadrature_interface,
																										const VectorTools::NormType												norm_type,
																										const ComponentMask														component_mask_domain,
																										const ComponentMask														component_mask_interface,
																										const double															exponent)
const
{
	return assembly_helper.compute_distance_to_other_solution(solution, other_incremental_fe.solution, other_incremental_fe.assembly_helper, quadrature_domain, quadrature_interface, norm_type, component_mask_domain, component_mask_interface, exponent);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
std::pair<const double, const double>
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::compute_distance_to_exact_solution(	const Function<spacedim>&		exact_solution_domain,
																										const Function<spacedim>&		exact_solution_interface,
																										const Quadrature<spacedim>		quadrature_domain,
																										const Quadrature<spacedim-1>	quadrature_interface,
																										const VectorTools::NormType		norm_type,
																										const ComponentMask				component_mask_domain,
																										const ComponentMask				component_mask_interface,
																										const double					exponent)
const
{
	return assembly_helper.compute_distance_to_exact_solution(solution, exact_solution_domain, exact_solution_interface, quadrature_domain, quadrature_interface, norm_type, component_mask_domain, component_mask_interface, exponent);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::read_solution_from_file(const string file_name)
{
    FILE *readin = fopen(file_name.c_str(),"r");
	Assert(readin != nullptr, ExcMessage("Could not open the file!"))
	for(unsigned int m = 0; m < solution.size(); ++m)
	{
		double temp_double;
		const unsigned int no_vals = fscanf(readin, "%le", &temp_double);
		(void) no_vals;	//silence unused parameter warnings of compiler
		Assert(no_vals == 1, ExcMessage("Error during reading results file. Is the format of the file correct?"));
		solution[m] = temp_double;
	}
    fclose(readin);
    zero_ghosts(solution);
	solution.compress(VectorOperation::insert);
	update_ghosts(solution);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::write_solution_to_file(const string file_name)
const
{
	FILE* printout = fopen (file_name.c_str(),"w");
	Assert(printout != nullptr, ExcMessage("Could not open the file!"));
	for(unsigned int m=0; m<solution.size(); ++m)
		fprintf(printout, "%- 1.16e\n", solution[m]);
	fclose(printout);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::write_output_independent_fields(	const string 		file_name_domain,
																									const string 		file_name_interface,
																									const unsigned int	n_subdivisions)
{
	const auto file_names = assembly_helper.write_output_independent_fields(solution,
																			file_name_domain,
																			file_name_interface,
																			global_data->get_time_step(),
																			dp_domain,
																			dp_interface,
																			n_subdivisions);

	const std::size_t pos_domain = file_names.first.find_last_of("/\\") + 1;
	const auto file_name_domain_wo_folder = file_names.first.substr(pos_domain);
	times_and_names_domain.push_back(make_pair(global_data->get_t(), file_name_domain_wo_folder));
	std::ofstream pvd_output_domain((file_name_domain + ".pvd").c_str());
	DataOutBase::write_pvd_record(pvd_output_domain, times_and_names_domain);

	const std::size_t pos_interface = file_names.second.find_last_of("/\\") + 1;
	const auto file_name_interface_wo_folder = file_names.second.substr(pos_interface);
	times_and_names_interface.push_back(make_pair(global_data->get_t(), file_name_interface_wo_folder));
	std::ofstream pvd_output_interface((file_name_interface + ".pvd").c_str());
	DataOutBase::write_pvd_record(pvd_output_interface, times_and_names_interface);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
const AssemblyHelper<spacedim>&
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::get_assembly_helper()
{
	return assembly_helper;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
double
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::get_potential_value()
const
{
	return potential_value;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::attach_data_postprocessor_domain(const dealii::DataPostprocessor<spacedim>& dp)
{
	dp_domain.push_back(&dp);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::attach_data_postprocessor_interface(const dealii::DataPostprocessor<spacedim>& dp)
{
	dp_interface.push_back(&dp);
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
const SolutionVectorType&
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::get_solution_vector()
const
{
	return solution;
}


template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::reinit_solution_vector(SolutionVectorType& vector)
{
	auto vector_ptr_sequential = dynamic_cast<Vector<double>*>(&vector);
#ifdef DEAL_II_WITH_MPI
	auto vector_ptr_parallel = dynamic_cast<LinearAlgebra::distributed::Vector<double>*>(&vector);
#endif // DEAL_II_WITH_MPI
	if(vector_ptr_sequential != nullptr)
		vector_ptr_sequential->reinit(assembly_helper.system_size());
#ifdef DEAL_II_WITH_MPI
	else if(vector_ptr_parallel != nullptr)
	{
		const auto tria_domain_ptr = dynamic_cast<const dealii::parallel::Triangulation<spacedim, spacedim>*>(&(assembly_helper.get_triangulation_system().get_triangulation_domain()));
		Assert(tria_domain_ptr != nullptr, ExcMessage("Internal error!"));
		vector_ptr_parallel->reinit(assembly_helper.get_locally_owned_indices(), assembly_helper.get_locally_relevant_indices(), tria_domain_ptr->get_communicator());
	}
#endif // DEAL_II_WITH_MPI
	else
	{
		Assert(false, ExcMessage("FEModel not used with an appropriate vector type for the solution!"));
	}
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::reinit_rhs_vector(RHSVectorType& vector)
{
	auto vector_ptr_sequential = dynamic_cast<BlockVector<double>*>(&vector);
#ifdef DEAL_II_WITH_PETSC
#ifdef DEAL_II_WITH_MPI
	auto vector_ptr_parallel = dynamic_cast<PETScWrappers::MPI::BlockVector*>(&vector);
#endif // DEAL_II_WITH_PETSC
#endif // DEAL_II_WITH_MPI
	auto index_sets = assembly_helper.get_locally_owned_indices_blocks();
	if(index_sets[1].size() == 0)
		index_sets.erase(index_sets.begin() + 1);
	if(index_sets[0].size() == 0)
		index_sets.erase(index_sets.begin());

	if(vector_ptr_sequential != nullptr)
	{
		const std::vector<unsigned int> block_sizes = {index_sets[0].size(), index_sets[1].size()};
		vector_ptr_sequential->reinit(block_sizes);
	}
#ifdef DEAL_II_WITH_PETSC
#ifdef DEAL_II_WITH_MPI
	else if(vector_ptr_parallel != nullptr)
	{
		const auto tria_domain_ptr = dynamic_cast<const dealii::parallel::Triangulation<spacedim, spacedim>*>(&(assembly_helper.get_triangulation_system().get_triangulation_domain()));
		Assert(tria_domain_ptr != nullptr, ExcMessage("Internal error!"));
		vector_ptr_parallel->reinit(index_sets, tria_domain_ptr->get_communicator());
	}
#endif // DEAL_II_WITH_PETSC
#endif // DEAL_II_WITH_MPI
	else
	{
		Assert(false, ExcMessage("FEModel not used with an appropriate block vector for the RHS!"));
	}

}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::reinit_matrix(MatrixType& vector)
{
	auto matrix_ptr_sequential = dynamic_cast<GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>*>(&vector);
#ifdef DEAL_II_WITH_PETSC
#ifdef DEAL_II_WITH_MPI
	auto matrix_ptr_parallel = dynamic_cast<GalerkinTools::parallel::TwoBlockMatrix<PETScWrappers::MPI::SparseMatrix>*>(&vector);
#endif // DEAL_II_WITH_PETSC
#endif // DEAL_II_WITH_MPI
	if(matrix_ptr_sequential != nullptr)
	{
		matrix_ptr_sequential->reinit(sparsity_pattern);
	}
#ifdef DEAL_II_WITH_PETSC
#ifdef DEAL_II_WITH_MPI
	else if(matrix_ptr_parallel != nullptr)
	{
		const auto tria_domain_ptr = dynamic_cast<const dealii::parallel::Triangulation<spacedim, spacedim>*>(&(assembly_helper.get_triangulation_system().get_triangulation_domain()));
		Assert(tria_domain_ptr != nullptr, ExcMessage("You are trying to perform a parallel computation with a sequential triangulation, which is not possible!"));
		matrix_ptr_parallel->reinit(sparsity_pattern, assembly_helper.get_locally_owned_indices(), tria_domain_ptr->get_communicator());
	}
#endif // DEAL_II_WITH_PETSC
#endif // DEAL_II_WITH_MPI
	else
	{
		Assert(false, ExcMessage("FEModel not used with an appropriate TwoBlockMatrix for the system matrix!"));
	}
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
bool
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::compute_system(	const SolutionVectorType& 	solution_ref,
																					AffineConstraints<double>&	constraints)
{
	// initialize system matrix, rhs vector, rhs scaling vector
	reinit_matrix(system_matrix);
	reinit_rhs_vector(rhs);

	// assemble system
	vector<const SolutionVectorType*> solution_ref_sets(1);
	solution_ref_sets[0] = &solution_ref;
	if(assembly_helper.assemble_system(	solution,
										solution_ref_sets,
										constraints,
										potential_value,
										rhs,
										system_matrix,
										make_tuple(true,true,true)))
	{
		return true;
	}
	else
	{
		// just to be sure
		constraints.set_zero(rhs);
		return false;
	}
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::compute_sparsity_pattern(const AffineConstraints<double>& constraints)
{
	sparsity_pattern.reinit(assembly_helper);
	assembly_helper.generate_sparsity_pattern_by_simulation(sparsity_pattern, constraints);
#ifdef DEAL_II_WITH_MPI
	const auto tria_domain_ptr = dynamic_cast<const dealii::parallel::Triangulation<spacedim, spacedim>*>(&(assembly_helper.get_triangulation_system().get_triangulation_domain()));
	if(tria_domain_ptr != nullptr)
		sparsity_pattern.distribute(assembly_helper.get_locally_owned_indices(), tria_domain_ptr->get_communicator());
#endif //DEAL_II_WITH_MPI
	sparsity_pattern.finalize();
	return;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::make_constraints(	AffineConstraints<double>&			constraints,
																					const AffineConstraints<double>&	custom_constraints,
																					const AffineConstraints<double>&	ignore_constraints)
{
	// note: hanging node constraints are not updated here because this is only necessary after mesh refinement

	// update Dirichlet constraints(before doing so, set up the constraint object all_ignore_constraints, which contains the lines
	// which are constrained already by hanging node constraints and those which are to be ignored anyway; no Dirichlet constraints will be
	// enforced for dofs related to lines included in all_ignore_constraints in order to avoid overconstraining);
	// also incorporate custom constraints into Dirichlet constraint object (but do not allow conflicts here)
	AffineConstraints<double> all_ignore_constraints(assembly_helper.get_locally_relevant_indices());
	all_ignore_constraints.merge(hanging_node_constraints, AffineConstraints<double>::MergeConflictBehavior::left_object_wins, true);
	all_ignore_constraints.merge(ignore_constraints, AffineConstraints<double>::MergeConflictBehavior::left_object_wins, true);
	all_ignore_constraints.close();
	dirichlet_constraints.reinit(assembly_helper.get_locally_relevant_indices());
	assembly_helper.make_dirichlet_constraints(	dirichlet_constraints,
												this->constraints->get_dirichlet_constraints(),
												all_ignore_constraints);
	dirichlet_constraints.close();
	dirichlet_constraints.merge(custom_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);

	// combine hanging node constraints and dirichlet constraints
	constraints.merge(hanging_node_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);
	constraints.merge(dirichlet_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);
	constraints.close();

	return;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::adjust_constraint_inhomogeneity(AffineConstraints<double>&	constraints)
const
{
	double inhomogeneity;
	for(const auto& line : dirichlet_constraints.get_lines())
	{
		const unsigned int constraint_index = line.index;
		inhomogeneity = line.inhomogeneity - solution[constraint_index];
		for(const auto& entry : line.entries)
			inhomogeneity += entry.second * solution[entry.first];
		constraints.set_inhomogeneity(constraint_index, inhomogeneity);
	}
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::update_rhs_scaling_vector()
{
	reinit_rhs_vector(rhs_scaling_vector);
	const auto index_sets = assembly_helper.get_locally_owned_indices_blocks();
	const auto& index_set_block_0 = index_sets[0];
	auto& system_matrix_A = system_matrix.get_A();
	auto& rhs_scaling_vector_block_0 = rhs_scaling_vector.block(0);

	if(index_set_block_0.n_elements() > 0)
	{
#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_PETSC)
		auto mtx_ptr = dynamic_cast<PETScWrappers::MPI::SparseMatrix*>(&system_matrix_A);
#else
		void* mtx_ptr = nullptr;
#endif // DEAL_II_WITH_MPI && DEAL_II_WITH_PETSC
		if(mtx_ptr == nullptr)
		{
			for(auto entry = system_matrix_A.begin(*index_set_block_0.begin()); entry != system_matrix_A.end( *index_set_block_0.begin() + index_set_block_0.n_elements() - 1 ); ++entry)
				if( fabs(entry->value()) > rhs_scaling_vector_block_0[entry->row()] )
					rhs_scaling_vector_block_0[entry->row()] = fabs(entry->value());
		}
		// if we deal with a PETSc matrix, use PETSc method directly because this is way faster
		else
		{
#if defined(DEAL_II_WITH_MPI) && defined(DEAL_II_WITH_PETSC)
			auto vct_ptr = dynamic_cast<PETScWrappers::MPI::Vector*>(&rhs_scaling_vector_block_0);
			Assert(vct_ptr != nullptr, ExcMessage("Vector type for scaling vector must be PETScWrappers::MPI::BlockVector!"));
			const PetscErrorCode ierr = MatGetRowMaxAbs(mtx_ptr->petsc_matrix(), *vct_ptr, nullptr);
			Assert(ierr == 0, ExcPETScError(ierr));
			(void)ierr;
#endif // DEAL_II_WITH_MPI && DEAL_II_WITH_PETSC
		}
		rhs_scaling_vector_block_0.compress(VectorOperation::insert);
	}
	return;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
double
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::compute_estimated_potential_increment(const SolutionVectorType& delta_solution)
const
{
	// compute estimated potential increment
	double estimated_potential_increment = 0.0;
	for(const auto locally_owned_index : assembly_helper.get_locally_owned_indices())
	estimated_potential_increment += delta_solution[locally_owned_index] * rhs[locally_owned_index];
#ifdef DEAL_II_WITH_MPI
	if(assembly_helper.get_triangulation_system().get_this_proc_n_procs().second > 1)
	{
		const auto tria_domain_ptr = dynamic_cast<const dealii::parallel::Triangulation<spacedim, spacedim>*>(&(assembly_helper.get_triangulation_system().get_triangulation_domain()));
		Assert(tria_domain_ptr != nullptr, ExcMessage("Internal error!"));
		double send_value = estimated_potential_increment;
		int ierr = MPI_Allreduce(&send_value, &estimated_potential_increment, 1, MPI_DOUBLE, MPI_SUM, tria_domain_ptr->get_communicator());
		AssertThrowMPI(ierr);
	}
#endif //DEAL_II_WITH_MPI
	return estimated_potential_increment;
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::adjust_delta_solution(SolutionVectorType& 				delta_solution,
																						const SolutionVectorType&			solution_ref,
																						const AffineConstraints<double>&	constraints)
{
	vector<const SolutionVectorType*> solution_ref_sets(1);
	solution_ref_sets[0] = &solution_ref;
	const double max_step = assembly_helper.get_maximum_step_length(solution, solution_ref_sets, delta_solution);
	if(max_step < 1.0/global_data->safety_distance)
	{
		const unsigned int this_proc = assembly_helper.get_triangulation_system().get_this_proc_n_procs().first;
		ConditionalOStream pout(cout, (this_proc == 0) && (global_data->get_output_level() > 0));
		pout << "CORRECTED STEP TO " << global_data->safety_distance * max_step << endl;
		delta_solution *= global_data->safety_distance * max_step;
		zero_ghosts(delta_solution);
		constraints.distribute(delta_solution);
		update_ghosts(delta_solution);
	}
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::update_ghosts(SolutionVectorType& vector)
{
	(void)vector;
#ifdef DEAL_II_WITH_MPI
	if(assembly_helper.get_triangulation_system().get_this_proc_n_procs().second > 1)
	{
		auto vct_ptr = dynamic_cast<LinearAlgebra::distributed::Vector<double>*>(&vector);
		if(vct_ptr != nullptr)
			vct_ptr->update_ghost_values();
		else
			Assert(false, ExcMessage("The vector type used for the solution vector is currently not supported by the function update_ghosts()!"));
	}
#endif //DEAL_II_WITH_MPI
}


template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::zero_ghosts(SolutionVectorType& vector)
{
	(void)vector;
#ifdef DEAL_II_WITH_MPI
	if(assembly_helper.get_triangulation_system().get_this_proc_n_procs().second > 1)
	{
		auto vct_ptr = dynamic_cast<LinearAlgebra::distributed::Vector<double>*>(&vector);
		if(vct_ptr != nullptr)
			vct_ptr->zero_out_ghosts();
		else
			Assert(false, ExcMessage("The vector type used for the solution vector is currently not supported by the function zero_ghosts()!"));
	}
#endif //DEAL_II_WITH_MPI
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
double
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::get_residual()
const
{
	// copy rhs before scaling
	// @todo Though this approach is simple, it is certainly not efficient!
	RHSVectorType scaled_rhs = rhs;

	// perform scaling
	// the scaling currently only applies to the first block of the system (i.e., the block related to FE dofs)
	auto& scaled_rhs_block_0 = scaled_rhs.block(0);
	const auto& rhs_scaling_vector_block_0 = rhs_scaling_vector.block(0);
	for(const auto& m : scaled_rhs_block_0.locally_owned_elements())
		scaled_rhs_block_0[m] = scaled_rhs_block_0[m] / rhs_scaling_vector_block_0[m];
	scaled_rhs_block_0.compress(VectorOperation::insert);

	// compute the residual
	double residual_0 = 0.0;
	double residual_1 = 0.0;
	if(scaled_rhs.block(0).size() > 0)
		residual_0 = scaled_rhs.block(0).l2_norm();
	if( (scaled_rhs.n_blocks() > 1) && (scaled_rhs.block(1).size() > 0) )
		residual_1 = scaled_rhs.block(1).l2_norm();

	return sqrt( (residual_0 * residual_0 + residual_1 * residual_1) / scaled_rhs.size());
}

template<unsigned int spacedim, class SolutionVectorType, class RHSVectorType, class MatrixType>
void
FEModel<spacedim, SolutionVectorType, RHSVectorType, MatrixType>::post_refinement()
{
	Assert(false, ExcMessage("Refinement is currently not implemented for the FEModel!"));
}

template class incrementalFE::FEModel<2, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>>;
template class incrementalFE::FEModel<3, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>>;

#ifdef DEAL_II_WITH_MPI
#ifdef DEAL_II_WITH_PETSC
	template class incrementalFE::FEModel<2, LinearAlgebra::distributed::Vector<double>, PETScWrappers::MPI::BlockVector, GalerkinTools::parallel::TwoBlockMatrix<PETScWrappers::MPI::SparseMatrix>>;
	template class incrementalFE::FEModel<3, LinearAlgebra::distributed::Vector<double>, PETScWrappers::MPI::BlockVector, GalerkinTools::parallel::TwoBlockMatrix<PETScWrappers::MPI::SparseMatrix>>;
#endif // DEAL_II_WITH_PETSC
#endif // DEAL_II_WITH_MPI
