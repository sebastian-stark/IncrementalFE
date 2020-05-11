#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/linear_material_00.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{
	//u
	IndependentField<spacedim-1, spacedim> u("u", FE_Nothing<spacedim-1, spacedim>(), 2, {});

	vector<DependentField<spacedim-1, spacedim>> dependent_fields;

	DependentField<spacedim-1, spacedim> u_1("u_1");
	u_1.add_term(1.0, u, 0);

	DependentField<spacedim-1, spacedim> u_2("u_2");
	u_2.add_term(1.0, u, 1);

	dependent_fields.push_back(u_1);
	dependent_fields.push_back(u_2);

	FullMatrix<double> C(2);
	for(unsigned int m = 0; m < C.m(); ++m)
		for(unsigned int n = 0; n < C.n(); ++n)
			C(m, n) = sqrt(1.0 + m + n);
	C.symmetrize();
	Vector<double> y(2);
	for(unsigned int m = 0; m < y.size(); ++m)
		y[m] = sqrt(1.0 + m);

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	LinearMaterial00<spacedim-1, spacedim> linear_material(	dependent_fields,
															{},
															QGauss<spacedim-1>(1),
															global_data,
															C,
															y,
															"",
															0.5);


	Vector<double> e_sigma(dependent_fields.size());
	vector<Vector<double>> e_sigma_ref_sets(1);
	e_sigma_ref_sets[0].reinit(e_sigma.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	Tensor<1, spacedim> n;
	const double N = e_sigma.size();
	for(unsigned int m = 0; m < e_sigma.size(); ++m)
	{
		e_sigma[m] = cos( 2.0*numbers::PI*((double)(m+1))/N ) + 1.1;
		e_sigma_ref_sets[0][m] = sin( 2.0*numbers::PI*((double)(m+1))/N ) + 1.1;
	}
	linear_material.compare_derivatives_with_numerical_derivatives(	e_sigma,
																	e_sigma_ref_sets,
																	hidden_vars,
																	x,
																	n);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
