#include <genmap-impl.h>

#define MM 505

int project(genmap_handle h, struct comm *gsc, mgData d, genmap_vector ri,
            int max_iter, genmap_vector x) {
  assert(x->size == ri->size);
  assert(x->size == genmap_get_nel(h));

  uint lelt = x->size;
  GenmapLong nelg = genmap_get_partition_nel(h);

  genmap_vector z0, z, dz, w, p, r;
  genmap_vector_create(&z, lelt);
  genmap_vector_create(&w, lelt);
  genmap_vector_create(&r, lelt);
  genmap_vector_create(&p, lelt);
  genmap_vector_create(&z0, lelt);
  genmap_vector_create(&dz, lelt);

  assert(max_iter < MM);
  double *P, *W;
  GenmapCalloc(lelt * MM, &P);
  GenmapCalloc(lelt * MM, &W);

  uint i;
  for (i = 0; i < lelt; i++) {
    x->data[i] = 0.0;
    r->data[i] = ri->data[i];
  }

  genmap_vector_copy(z, r);
  genmap_vector_copy(p, z);

  GenmapScalar rz1 = genmap_vector_dot(r, z), buf;
  comm_allreduce(gsc, gs_double, gs_add, &rz1, 1, &buf);

  GenmapScalar rr = genmap_vector_dot(r, r);
  comm_allreduce(gsc, gs_double, gs_add, &rr, 1, &buf);

  GenmapScalar alpha, beta, rz0, rz2, scale;

  double tol = 1e-5;
  double res_tol = rr * tol;

  uint j, k;
  i = 0;
  while (i < max_iter) {
    metric_tic(gsc, LAPLACIAN);
#if 1
    GenmapLaplacianWeighted(h, p->data, w->data);
#else
    GenmapLaplacian(h, p->data, w->data);
#endif
    metric_toc(gsc, LAPLACIAN);

    GenmapScalar den = genmap_vector_dot(p, w);
    comm_allreduce(gsc, gs_double, gs_add, &den, 1, &buf);
    alpha = rz1 / den;

    scale = 1.0 / sqrt(den);
    for (j = 0; j < lelt; j++) {
      W[i * lelt + j] = scale * w->data[j];
      P[i * lelt + j] = scale * p->data[j];
    }

    genmap_vector_axpby(x, x, 1.0, p, alpha);
    genmap_vector_axpby(r, r, 1.0, w, -alpha);

    rr = genmap_vector_dot(r, r);
    comm_allreduce(gsc, gs_double, gs_add, &rr, 1, &buf);

    if (rr < res_tol || sqrt(rr) < tol)
      break;

    GenmapScalar norm0 = genmap_vector_dot(z, z);
    comm_allreduce(gsc, gs_double, gs_add, &norm0, 1, &buf);

    genmap_vector_copy(z0, z);
    mg_vcycle(z->data, r->data, d);

    GenmapScalar norm1 = genmap_vector_dot(z, z);
    comm_allreduce(gsc, gs_double, gs_add, &norm1, 1, &buf);

    rz0 = rz1;
    genmap_vector_ortho_one(gsc, z, nelg);
    rz1 = genmap_vector_dot(r, z);
    comm_allreduce(gsc, gs_double, gs_add, &rz1, 1, &buf);

    genmap_vector_axpby(dz, z, 1.0, z0, -1.0);
    rz2 = genmap_vector_dot(r, dz);
    comm_allreduce(gsc, gs_double, gs_add, &rz2, 1, &buf);

    beta = rz2 / rz0;
    genmap_vector_axpby(p, z, 1.0, p, beta);

    i++;

    metric_tic(gsc, PROJECT);
    for (k = 0; k < lelt; k++)
      P[(MM - 1) * lelt + k] = 0.0;

    for (j = 0; j < i; j++) {
      double a = 0.0;
      for (k = 0; k < lelt; k++)
        a += W[j * lelt + k] * p->data[k];
      comm_allreduce(gsc, gs_double, gs_add, &a, 1, &buf);
      for (k = 0; k < lelt; k++)
        P[(MM - 1) * lelt + k] += a * P[j * lelt + k];
    }

    for (k = 0; k < lelt; k++)
      p->data[k] -= P[(MM - 1) * lelt + k];
    metric_toc(gsc, PROJECT);
  }

  GenmapDestroyVector(z);
  GenmapDestroyVector(w);
  GenmapDestroyVector(p);
  GenmapDestroyVector(r);
  GenmapDestroyVector(z0);
  GenmapDestroyVector(dz);

  GenmapFree(P);
  GenmapFree(W);

  return i + 1;
}
