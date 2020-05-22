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
#include <time.h>
#include <stdlib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/omega.h>
#include <incremental_fe/scalar_functionals/dissipation_species_flux_00.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
class DissipationSpeciesFluxOmega00 : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	const double D;

public:

	DissipationSpeciesFluxOmega00(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
									const std::set<dealii::types::material_id>									domain_of_integration,
									const dealii::Quadrature<spacedim>											quadrature,
									GlobalDataIncrementalFE<spacedim>&											global_data,
									const double																D,
									const unsigned int															method,
									const double																alpha = 0.0,
									const std::string															name = "Omega")
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 3, 0, 0, 1, method, alpha, name),
	D(D)
	{
	}

	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{
		Tensor<1, 3> i;
		for(unsigned int m = 0; m < 3; ++m)
			i[m] = values[m];
		const double q = values[3];

		if(q <= 0.0)
			return true;

		if(get<0>(requested_quantities))
		{
			omega = 1.0 / ( 2.0 * D * q) * i * i;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				d_omega[m] = 1.0 / ( D * q ) * i[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 3; ++m)
				d2_omega(m, m) = 1.0 / ( D * q );

			if(compute_dq)
			{
				for(unsigned int m = 0; m < 3; ++m)
					d2_omega(m, 3) = - 1.0 / ( D * q * q ) * i[m];
			}
		}

		return false;
	}

};


void test(const unsigned int method, const bool predictor)
{
	const unsigned int spacedim = 3;

	srand(time(NULL));

	vector<DependentField<spacedim, spacedim>> dependent_fields(4, DependentField<spacedim, spacedim>("q"));

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	if(method == 2)
		global_data.set_predictor_corrector(true);
	global_data.set_predictor_step(predictor);

	const double alpha = 0.5;
	const double t0 = 2.0;
	const double t1 = 2.5;
	const double D = 1.3;

	DissipationSpeciesFluxOmega00<spacedim> omega_domain(	dependent_fields,
															{},
															QGauss<spacedim>(1),
															global_data,
															D,
															method,
															alpha);

	DissipationSpeciesFlux00<spacedim> omega_domain_2(	dependent_fields,
														{},
														QGauss<spacedim>(1),
														global_data,
														D,
														0.0,
														alpha,
														method,
														0);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	for(unsigned int m = 0; m < dependent_fields.size(); ++m)
	{
		e_omega[m] = (double)rand() / RAND_MAX;
		e_omega_ref_sets[0][m] = (double)rand() / RAND_MAX;
	}
	global_data.set_t(t0);
	global_data.set_t(t1);

	Point<spacedim> x;
	Vector<double> hidden_vars(1);
	for(unsigned int m = 0; m < 1; ++m)
		hidden_vars[m] = (double)rand() / RAND_MAX;
	Vector<double> hidden_vars_2;
	hidden_vars_2 = hidden_vars;

	double d;
	Vector<double> d1(4);
	FullMatrix<double> d2(4,4);

	omega_domain.get_h_omega(e_omega, e_omega_ref_sets, hidden_vars, x, d, d1, d2, make_tuple(true, true, true));

	double d_;
	Vector<double> d1_(4);
	FullMatrix<double> d2_(4,4);

	omega_domain_2.get_h_omega(e_omega, e_omega_ref_sets, hidden_vars_2, x, d_, d1_, d2_, make_tuple(true, true, true));

	double e = 0.0;
	e += fabs(d - d_);
	for(unsigned int m = 0; m < dependent_fields.size(); ++m)
		e += fabs(d1[m] - d1_[m]);
	for(unsigned int m = 0; m < dependent_fields.size(); ++m)
		for(unsigned int n = 0; n < dependent_fields.size(); ++n)
			e += fabs(d2(m, n) - d2_(m, n));
	cout << e << endl;
}

int main()
{
	test(0, true);
	test(1, true);
	test(2, true);
	test(2, false);

	return 0;
}
