#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/neo_hooke_02.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{

	vector<DependentField<spacedim, spacedim>> dependent_fields(9, DependentField<spacedim, spacedim>("q"));

	const double lambda = 0.5;
	const double mu = 0.7;
	const double n_0 = 0.3;
	const double alpha = 0.5;
	ConstantFunction<spacedim> fun(1.3);

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	NeoHooke02<spacedim> neo_hooke_01(	dependent_fields,
										{},
										QGauss<spacedim>(1),
										global_data,
										lambda,
										mu,
										alpha,
										fun);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;

	e_omega[0] = -0.151440;
	e_omega[1] = 0.104478;
	e_omega[2] = -0.918154;
	e_omega[3] = -0.389247;
	e_omega[4] = -0.996425;
	e_omega[5] = 0.038264;
	e_omega[6] = -0.586984;
	e_omega[7] = 0.633806;
	e_omega[8] = 0.211506;

	e_omega_ref_sets[0][0] = -0.27231;
	e_omega_ref_sets[0][1] = -0.13911;
	e_omega_ref_sets[0][2] = -0.55068;
	e_omega_ref_sets[0][3] = -0.30054;
	e_omega_ref_sets[0][4] = -0.39964;
	e_omega_ref_sets[0][5] = 0.35431;
	e_omega_ref_sets[0][6] = -0.32508;
	e_omega_ref_sets[0][7] = 0.48600;
	e_omega_ref_sets[0][8] = 0.13373;

	neo_hooke_01.compare_derivatives_with_numerical_derivatives(e_omega,
																e_omega_ref_sets,
																hidden_vars,
																x);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
