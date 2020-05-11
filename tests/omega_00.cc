#include <iostream>
#include <time.h>
#include <stdlib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/omega.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
class OmegaDomain : public incrementalFE::Omega<spacedim, spacedim>
{

private:

	FullMatrix<double>
	A;

	Vector<double>
	b;

public:

	OmegaDomain(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>									domain_of_integration,
					const dealii::Quadrature<spacedim>											quadrature,
					GlobalDataIncrementalFE<spacedim>&											global_data,
					const FullMatrix<double>													A,
					const Vector<double>														b,
					const unsigned int															method,
					const double																alpha = 0.0,
					const std::string															name = "Omega")
	:
	Omega<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, 4, 2, 3, 3, method, alpha, name),
	A(A),
	b(b)
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
		Vector<double> A_v(12);
		A.vmult(A_v, values);

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * (values * A_v ) + b * values;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = A_v[m] + b[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < (compute_dq ? 12 : 9); ++n)
					d2_omega(m, n) = A(m, n);
		}

		return false;
	}

};

template<unsigned int spacedim>
class OmegaInterface : public incrementalFE::Omega<spacedim-1, spacedim>
{

private:

	FullMatrix<double>
	A;

	Vector<double>
	b;

public:

	OmegaInterface(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_sigma,
					const std::set<dealii::types::material_id>										domain_of_integration,
					const dealii::Quadrature<spacedim-1>											quadrature,
					GlobalDataIncrementalFE<spacedim>&												global_data,
					const FullMatrix<double>														A,
					const Vector<double>															b,
					const unsigned int																method,
					const double																	alpha = 0.0,
					const std::string																name = "Omega")
	:
	Omega<spacedim-1, spacedim>(e_sigma, domain_of_integration, quadrature, global_data, 4, 2, 3, 3, method, alpha, name),
	A(A),
	b(b)
	{
	}

	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const double						/*t*/,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1,spacedim>& 	/*n*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities,
								const bool							compute_dq)
	const
	{
		Vector<double> A_v(12);
		A.vmult(A_v, values);

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * (values * A_v ) + b * values;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				d_omega[m] = A_v[m] + b[m];
		}

		if(get<2>(requested_quantities))
		{
			for(unsigned int m = 0; m < 9; ++m)
				for(unsigned int n = 0; n < (compute_dq ? 12 : 9); ++n)
					d2_omega(m, n) = A(m, n);
		}

		return false;
	}

};

void test_domain(const unsigned int method, const bool predictor)
{
	const unsigned int spacedim = 3;

	srand(time(NULL));

	vector<DependentField<spacedim, spacedim>> dependent_fields(12, DependentField<spacedim, spacedim>("q"));

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	if(method == 2)
		global_data.set_predictor_corrector(true);
	global_data.set_predictor_step(predictor);

	const double alpha = 0.5;
	const double t0 = 2.0;
	const double t1 = 2.5;

	FullMatrix<double> A(12,12);
	Vector<double> b(12);
	for(unsigned int m = 0; m < 12; ++m)
	{
		b(m) = (double)rand() / RAND_MAX;
		for(unsigned int n = m; n < 12; ++n)
			A(m, n) = A(n, m) = (double)rand() / RAND_MAX;
	}

	OmegaDomain<spacedim> omega_domain(	dependent_fields,
										{},
										QGauss<spacedim>(1),
										global_data,
										A,
										b,
										method,
										alpha);

	Vector<double> e_omega(12);
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(12);
	for(unsigned int m = 0; m < 12; ++m)
	{
		e_omega[m] = (double)rand() / RAND_MAX;
		e_omega_ref_sets[0][m] = (double)rand() / RAND_MAX;
	}
	global_data.set_t(t0);
	global_data.set_t(t1);

	Point<spacedim> x;
	Vector<double> hidden_vars(3);
	for(unsigned int m = 0; m < 3; ++m)
	{
		hidden_vars[m] = (double)rand() / RAND_MAX;
	}

	double d;
	Vector<double> d1(12);
	FullMatrix<double> d2(12,12);

	omega_domain.get_h_omega(e_omega, e_omega_ref_sets, hidden_vars, x, d, d1, d2, make_tuple(true, true, true));

	Vector<double> values(12);
	for(unsigned int m = 0; m < 6; ++m)
		values[m] = (e_omega[m] - e_omega_ref_sets[0][m])/(t1 - t0);
	for(unsigned int m = 0; m < 3; ++m)
		values[m + 6] = e_omega[m + 6];
	for(unsigned int m = 0; m < 3; ++m)
	{
		if(method == 0)
			values[m + 9] = e_omega_ref_sets[0][m + 9];
		else if(method == 1)
			values[m + 9] = alpha * e_omega[m + 9] + (1.0 - alpha) * e_omega_ref_sets[0][m + 9];
		else if(method == 2)
		{
			if(predictor)
				values[m + 9] = e_omega_ref_sets[0][m + 9];
			else
				values[m + 9] = alpha * hidden_vars[m] + (1.0 - alpha) * e_omega_ref_sets[0][m + 9];
		}
	}
	FullMatrix<double> I(12, 12);
	for(unsigned int m = 0; m < 12; ++m)
	{
		if(m < 6)
			I(m, m) = 1.0/(t1 - t0);
		else
			I(m, m) = 1.0;
	}

	FullMatrix<double> I_A(12, 12), I_A_I(12, 12);
	Vector<double> I_b(12);
	I.mmult(I_A, A);
	I.vmult(I_b, b);
	I_A.mTmult(I_A_I, I);

	double d_ = 0.0;
	Vector<double> d1_(12);
	FullMatrix<double> d2_(12,12);
	Vector<double> A_v(12);
	A.vmult(A_v, values);

	if(method != 1)
		d_ = (t1 - t0) * (0.5 * (A_v * values) + (b *values));
	Vector<double> A_v_b(12);
	for(unsigned int m = 0; m < 9; ++m)
		A_v_b[m] = A_v[m] + b[m];
	I.vmult(d1_, A_v_b);
	d1_ *= (t1 - t0);

	double e = 0.0;
	e += fabs(d - d_);
	for(unsigned int m = 0; m < 12; ++m)
		e += fabs(d1[m] - d1_[m]);
	d2_ = I_A_I;
	for(unsigned int m = 9; m < 12; ++m)
		for(unsigned int n = 0; n < 12; ++n)
			d2_(m, n) = 0.0;
	for(unsigned int m = 9; m < 12; ++m)
		for(unsigned int n = 0; n < 12; ++n)
			d2_(n, m) = method == 1 ? d2_(n, m) * alpha : 0.0;
	d2_ *= (t1 - t0);
	for(unsigned int m = 0; m < 12; ++m)
		for(unsigned int n = 0; n < 12; ++n)
			e += fabs(d2(m,n) - d2_(m,n));
	cout << e << endl;

}

void test_interface(const unsigned int method, const bool predictor)
{
	const unsigned int spacedim = 3;

	srand(time(NULL));

	vector<DependentField<spacedim-1, spacedim>> dependent_fields(12, DependentField<spacedim-1, spacedim>("q"));

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);
	if(method == 2)
		global_data.set_predictor_corrector(true);
	global_data.set_predictor_step(predictor);

	const double alpha = 0.5;
	const double t0 = 2.0;
	const double t1 = 2.5;

	FullMatrix<double> A(12,12);
	Vector<double> b(12);
	for(unsigned int m = 0; m < 12; ++m)
	{
		b(m) = (double)rand() / RAND_MAX;
		for(unsigned int n = m; n < 12; ++n)
			A(m, n) = A(n, m) = (double)rand() / RAND_MAX;
	}

	OmegaInterface<spacedim> omega_interface(	dependent_fields,
											{},
											QGauss<spacedim-1>(1),
											global_data,
											A,
											b,
											method,
											alpha);

	Vector<double> e_omega(12);
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(12);
	for(unsigned int m = 0; m < 12; ++m)
	{
		e_omega[m] = (double)rand() / RAND_MAX;
		e_omega_ref_sets[0][m] = (double)rand() / RAND_MAX;
	}
	global_data.set_t(t0);
	global_data.set_t(t1);

	Point<spacedim> x;
	Vector<double> hidden_vars(3);
	for(unsigned int m = 0; m < 3; ++m)
	{
		hidden_vars[m] = (double)rand() / RAND_MAX;
	}

	double d;
	Vector<double> d1(12);
	FullMatrix<double> d2(12,12);
	Tensor<1, spacedim> n;
	omega_interface.get_h_sigma(e_omega, e_omega_ref_sets, hidden_vars, x, n, d, d1, d2, make_tuple(true, true, true));

	Vector<double> values(12);
	for(unsigned int m = 0; m < 6; ++m)
		values[m] = (e_omega[m] - e_omega_ref_sets[0][m])/(t1 - t0);
	for(unsigned int m = 0; m < 3; ++m)
		values[m + 6] = e_omega[m + 6];
	for(unsigned int m = 0; m < 3; ++m)
	{
		if(method == 0)
			values[m + 9] = e_omega_ref_sets[0][m + 9];
		else if(method == 1)
			values[m + 9] = alpha * e_omega[m + 9] + (1.0 - alpha) * e_omega_ref_sets[0][m + 9];
		else if(method == 2)
		{
			if(predictor)
				values[m + 9] = e_omega_ref_sets[0][m + 9];
			else
				values[m + 9] = alpha * hidden_vars[m] + (1.0 - alpha) * e_omega_ref_sets[0][m + 9];
		}
	}
	FullMatrix<double> I(12, 12);
	for(unsigned int m = 0; m < 12; ++m)
	{
		if(m < 6)
			I(m, m) = 1.0/(t1 - t0);
		else
			I(m, m) = 1.0;
	}

	FullMatrix<double> I_A(12, 12), I_A_I(12, 12);
	Vector<double> I_b(12);
	I.mmult(I_A, A);
	I.vmult(I_b, b);
	I_A.mTmult(I_A_I, I);

	double d_ = 0.0;
	Vector<double> d1_(12);
	FullMatrix<double> d2_(12,12);
	Vector<double> A_v(12);
	A.vmult(A_v, values);

	if(method != 1)
		d_ = (t1 - t0) * (0.5 * (A_v * values) + (b *values));
	Vector<double> A_v_b(12);
	for(unsigned int m = 0; m < 9; ++m)
		A_v_b[m] = A_v[m] + b[m];
	I.vmult(d1_, A_v_b);
	d1_ *= (t1 - t0);

	double e = 0.0;
	e += fabs(d - d_);
	for(unsigned int m = 0; m < 12; ++m)
		e += fabs(d1[m] - d1_[m]);
	d2_ = I_A_I;
	for(unsigned int m = 9; m < 12; ++m)
		for(unsigned int n = 0; n < 12; ++n)
			d2_(m, n) = 0.0;
	for(unsigned int m = 9; m < 12; ++m)
		for(unsigned int n = 0; n < 12; ++n)
			d2_(n, m) = method == 1 ? d2_(n, m) * alpha : 0.0;
	d2_ *= (t1 - t0);
	for(unsigned int m = 0; m < 12; ++m)
		for(unsigned int n = 0; n < 12; ++n)
			e += fabs(d2(m,n) - d2_(m,n));
	cout << e << endl;

}

int main()
{
	test_domain(0, true);
	test_domain(1, true);
	test_domain(2, true);
	test_domain(2, false);
	test_interface(0, true);
	test_interface(1, true);
	test_interface(2, true);
	test_interface(2, false);

	return 0;
}
