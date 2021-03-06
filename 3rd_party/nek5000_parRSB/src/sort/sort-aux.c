#include <float.h>
#include <sort-impl.h>

double get_scalar(struct array *a, uint i, uint offset, uint usize,
                  gs_dom type) {
  char *v = (char *)a->ptr + i * usize + offset;

  double data;
  switch (type) {
  case gs_int:
    data = *((uint *)v);
    break;
  case gs_long:
    data = *((ulong *)v);
    break;
  case gs_double:
    data = *((double *)v);
    break;
  default:
    break;
  }

  return data;
}

void get_extrema(void *extrema_, struct sort *data, uint field,
                 struct comm *c) {
  struct array *a = data->a;
  uint usize = data->unit_size;
  uint offset = data->offset[field];
  gs_dom t = data->t[field];

  double *extrema = extrema_;
  sint size = a->n;
  if (size == 0) {
    extrema[0] = -DBL_MAX;
    extrema[1] = -DBL_MAX;
  } else {
    extrema[0] = -get_scalar(a, 0, offset, usize, t);
    extrema[1] = get_scalar(a, size - 1, offset, usize, t);
  }

  double buf[2];
  comm_allreduce(c, gs_double, gs_max, extrema, 2, buf);
  extrema[0] *= -1;
}

int set_dest(uint *proc, uint size, sint np, slong start, slong nelem) {
  uint i;
  if (nelem < np) {
    for (i = 0; i < size; i++)
      proc[i] = start + i;
    return 0;
  }

  uint psize = nelem / np;
  uint nrem = nelem - np * psize;
  slong lower = nrem * (psize + 1);

  uint id1, id2;
  if (start < lower)
    id1 = start / (psize + 1);
  else
    id1 = nrem + (start - lower) / psize;

  if (start + size < lower)
    id2 = (start + size) / (psize + 1);
  else
    id2 = nrem + (start + size - lower) / psize;

  i = 0;
  while (id1 <= id2 && i < size) {
    ulong s = id1 * psize + min(id1, nrem);
    ulong e = (id1 + 1) * psize + min(id1 + 1, nrem);
    e = min(e, nelem);
    assert(id1 < np && "Ooops ! id1 is larger than np");
    assert(id1 >= 0 && "Ooops ! id1 is smaller than 0");
    while (s <= start + i && start + i < e && i < size)
      proc[i++] = id1;
    id1++;
  }

  return 0;
}

int load_balance(struct array *a, size_t size, struct comm *c,
                 struct crystal *cr) {
  slong out[2][1], buf[2][1];
  slong in = a->n;
  comm_scan(out, c, gs_long, gs_add, &in, 1, buf);
  slong start = out[0][0];
  slong nelem = out[1][0];

  uint *proc;
  GenmapCalloc(a->n, &proc);
  set_dest(proc, a->n, c->np, start, nelem);

  sarray_transfer_ext_(a, size, proc, sizeof(uint), cr);

  GenmapFree(proc);

  return 0;
}
