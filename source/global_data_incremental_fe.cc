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

#include <incremental_fe/global_data_incremental_fe.h>

#include <iostream>
#include <deal.II/base/utilities.h>

using namespace std;
using namespace dealii;
using namespace incrementalFE;

template<unsigned int spacedim>
GlobalDataIncrementalFE<spacedim>::GlobalDataIncrementalFE(const double t_init)
:
t(t_init),
t_ref(t_init),
t_ref_old(t_init)
{
}

template<unsigned int spacedim>
GlobalDataIncrementalFE<spacedim>::~GlobalDataIncrementalFE()
{
	Assert(n_subscriptions() == 0, ExcMessage("You are about to destroy a GlobalDataIncrementalFE, which is currently in use! Make sure that all GlobalDataIncrementalFE objects live at least as long as the objects using them!"));
}

template<unsigned int spacedim>
double
GlobalDataIncrementalFE<spacedim>::get_t()
const
{
	return t;
}

template<unsigned int spacedim>
double
GlobalDataIncrementalFE<spacedim>::get_t_ref()
const
{
	return t_ref;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_t(const double t)
{
	Assert(t >= this->t, ExcMessage("The new time must be larger than the old one!"));
	t_ref_old = this->t_ref;
	t_ref = this->t;
	this->t = t;
	time_step++;
	reset_t_possible = true;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::reset_t()
{
	Assert(reset_t_possible, ExcMessage("You can reset the time only once because only the previous state is stored!"));
	t = t_ref;
	t_ref = t_ref_old;
	time_step--;
	reset_t_possible = false;
}

template<unsigned int spacedim>
unsigned int
GlobalDataIncrementalFE<spacedim>::get_time_step()
const
{
	return time_step;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::write_error_message(const string error_message)
{
	error_messages.push_back(error_message);
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::write_error_message(	const string			error_message,
														const Point<spacedim>&	location)
{

	string error_message_ = error_message + " at point (";
	for(unsigned int coordinate = 0; coordinate < spacedim-1; ++coordinate)
		error_message_ = error_message_ + Utilities::to_string(location(coordinate)) + ", ";
	error_message_ = error_message_ + Utilities::to_string(location(spacedim-1)) + ")";
	error_messages.push_back(error_message_);
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::print_error_messages()
const
{
	cout << "Error messages:\n";
	for(const auto& error_message: error_messages)
		cout << "  " << error_message << endl;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::print_last_error_message()
const
{
	cout << "Last error message:\n";
	cout << "  " << error_messages.back() << endl;
}


template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_sym_mode(const bool sym_mode)
{
	this->sym_mode = sym_mode;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_max_iter(const unsigned int max_iter)
{
	this->max_iter = max_iter;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_max_cutbacks(const unsigned int max_cutbacks)
{
	this->max_cutbacks = max_cutbacks;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_threshold_potential_increment(const double threshold_potential_increment)
{
	this->threshold_potential_increment = threshold_potential_increment;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_threshold_residual(const double threshold_residual)
{
	this->threshold_residual = threshold_residual;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_threshold_step_size(const double threshold_step_size)
{
	this->threshold_step_size = threshold_step_size;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_force_linear(const bool force_linear)
{
	this->force_linear = force_linear;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_predictor_corrector(const bool predictor_corrector)
{
	this->predictor_corrector = predictor_corrector;
}

template<unsigned int spacedim>
bool
GlobalDataIncrementalFE<spacedim>::get_predictor_step()
const
{
	return predictor_step;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_predictor_step(const bool predictor_step)
{
	this->predictor_step = predictor_step;
}

template<unsigned int spacedim>
unsigned int
GlobalDataIncrementalFE<spacedim>::get_output_level()
const
{
	return output_level;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_output_level(const unsigned int output_level)
{
	Assert(output_level < 2, ExcMessage("Output level can only be 0 or 1"));
	this->output_level = output_level;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_safety_distance(const double safety_distance)
{
	Assert(safety_distance < 1.0, ExcMessage("Safety distance must be smaller than 1.0"));
	Assert(safety_distance > 0.0, ExcMessage("Safety must be larger than 0.0"));
	this->safety_distance = safety_distance;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_perform_line_search(const bool perform_line_search)
{
	this->perform_line_search = perform_line_search;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_compute_sparsity_pattern(const unsigned int compute_sparsity_pattern)
{
	this->compute_sparsity_pattern = compute_sparsity_pattern;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::reinit(const double t_init)
{
	t = t_init;
	t_ref = t_init;
	t_ref_old = t_init;
	time_step = 0;
	reset_t_possible = false;
	error_messages.clear();
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_not_converged_at_local_level()
{
	converged_at_local_level = false;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_use_previous_increment_for_initial_guess(const bool use_previous_increment_for_initial_guess)
{
	this->use_previous_increment_for_initial_guess = use_previous_increment_for_initial_guess;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_scale_residual(const bool scale_residual)
{
	this->scale_residual = scale_residual;
}

template<unsigned int spacedim>
void
GlobalDataIncrementalFE<spacedim>::set_continue_on_nonconvergence(const bool continue_on_nonconvergence)
{
	this->continue_on_nonconvergence = continue_on_nonconvergence;
}

template<unsigned int spacedim>
bool
GlobalDataIncrementalFE<spacedim>::get_use_manufactured_solution()
const
{
	return use_manufactured_solution;
}

template class incrementalFE::GlobalDataIncrementalFE<2>;
template class incrementalFE::GlobalDataIncrementalFE<3>;
