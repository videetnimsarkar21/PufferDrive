#include "park.h"
#include "planner.h"

#include <Python.h>
#include <numpy/arrayobject.h>

#define PARK_ERROR_MESSAGE_CAPACITY 512
#define PARK_ENV_INIT_POSITIONAL_ARGS 7
#define PARK_AGENT_STATE_FIELDS 12
#define PARK_PLAN_FIELDS 5
#define PARK_SEGMENT_FIELDS 4

typedef struct {
    Park **envs;
    int num_envs;
} ParkVecEnv;

static int park_kwarg_float(PyObject *kwargs, const char *key, float *out) {
    PyObject *value = PyDict_GetItemString(kwargs, key);
    if (value == NULL) {
        PyErr_Format(PyExc_TypeError, "missing required keyword argument '%s'", key);
        return -1;
    }
    double parsed = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    if (!isfinite(parsed)) {
        PyErr_Format(PyExc_ValueError, "keyword argument '%s' must be finite", key);
        return -1;
    }
    *out = (float) parsed;
    return 0;
}

static int park_kwarg_int(PyObject *kwargs, const char *key, int *out) {
    PyObject *value = PyDict_GetItemString(kwargs, key);
    if (value == NULL) {
        PyErr_Format(PyExc_TypeError, "missing required keyword argument '%s'", key);
        return -1;
    }
    long parsed = PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        return -1;
    }
    if (parsed < INT_MIN || parsed > INT_MAX) {
        PyErr_Format(PyExc_ValueError, "keyword argument '%s' is out of int range", key);
        return -1;
    }
    *out = (int) parsed;
    return 0;
}

typedef struct {
    const char *key;
    float *value;
} ParkFloatField;

typedef struct {
    const char *key;
    int *value;
} ParkIntField;

static int park_parse_config(PyObject *kwargs, ParkConfig *config) {
    const ParkFloatField float_fields[] = {
        {"dt", &config->dt_s},
        {"goal_weight_min", &config->goal_weight_min},
        {"goal_weight_max", &config->goal_weight_max},
        {"collision_weight_min", &config->collision_weight_min},
        {"collision_weight_max", &config->collision_weight_max},
        {"goal_weight_default", &config->goal_weight_default},
        {"collision_weight_default", &config->collision_weight_default},
        {"reward_success", &config->reward_success},
        {"reward_collision", &config->reward_collision},
        {"reward_offroad", &config->reward_offroad},
        {"reward_handoff", &config->reward_handoff},
        {"threat_penalty_weight", &config->threat_penalty_weight},
        {"reward_static_clearance", &config->reward_static_clearance},
        {"reward_route_progress_per_m", &config->reward_route_progress_per_m},
        {"reward_subgoal", &config->reward_subgoal},
        {"reward_speed_target", &config->reward_speed_target},
        {"speed_target_mps", &config->speed_target_mps},
        {"speed_target_band_mps", &config->speed_target_band_mps},
        {"reward_plan_progress_per_m", &config->reward_plan_progress_per_m},
        {"imitation_weight", &config->imitation_weight},
        {"imitation_scale", &config->imitation_scale},
        {"refine_gain", &config->refine_gain},
        {"refine_hold_bonus", &config->refine_hold_bonus},
        {"refine_tanh_scale_m", &config->refine_tanh_scale_m},
        {"threat_decay_m", &config->threat_decay_m},
        {"threat_squash_mps", &config->threat_squash_mps},
        {"trust_cross_track_scale_m", &config->trust_cross_track_scale_m},
        {"trust_clearance_slope_per_m", &config->trust_clearance_slope_per_m},
        {"trust_clearance_threshold_m", &config->trust_clearance_threshold_m},
        {"success_position_tolerance_m", &config->success_position_tolerance_m},
        {"success_heading_tolerance_rad", &config->success_heading_tolerance_rad},
        {"success_speed_tolerance_mps", &config->success_speed_tolerance_mps},
        {"tier_weight_sparse", &config->tier_weights[0]},
        {"tier_weight_medium", &config->tier_weights[1]},
        {"tier_weight_dense", &config->tier_weights[2]},
    };
    const ParkIntField int_fields[] = {
        {"episode_steps", &config->episode_steps},
        {"max_agents", &config->max_agents},
        {"max_controlled_per_scene", &config->max_controlled_per_scene},
        {"obs_partner_slots", &config->partner_slots},
        {"obs_lane_slots", &config->lane_slots},
        {"obs_boundary_slots", &config->boundary_slots},
        {"reward_conditioning", &config->reward_conditioning},
        {"success_hold_steps", &config->success_hold_steps},
        {"refine_budget_steps", &config->refine_budget_steps},
        {"curriculum_ramp_steps", &config->curriculum_ramp_steps},
        {"stagger_first_episode", &config->stagger_first_episode},
    };
    for (size_t field_idx = 0; field_idx < sizeof(float_fields) / sizeof(float_fields[0]); field_idx++) {
        if (park_kwarg_float(kwargs, float_fields[field_idx].key, float_fields[field_idx].value) != 0) {
            return -1;
        }
    }
    for (size_t field_idx = 0; field_idx < sizeof(int_fields) / sizeof(int_fields[0]); field_idx++) {
        if (park_kwarg_int(kwargs, int_fields[field_idx].key, int_fields[field_idx].value) != 0) {
            return -1;
        }
    }
    return 0;
}

static void *park_array_data(PyObject *object, const char *name, int dtype, npy_intp element_count) {
    if (!PyArray_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name);
        return NULL;
    }
    PyArrayObject *array = (PyArrayObject *) object;
    if (!PyArray_IS_C_CONTIGUOUS(array) || PyArray_TYPE(array) != dtype || PyArray_SIZE(array) != element_count) {
        PyErr_Format(
            PyExc_ValueError,
            "%s must be a contiguous array of %ld elements with the expected dtype",
            name,
            (long) element_count);
        return NULL;
    }
    return PyArray_DATA(array);
}

static int park_bind_buffers(Park *env, PyObject *args) {
    npy_intp agent_count = env->config.max_agents;
    npy_intp observation_count = agent_count * env->observation_size;
    env->observations = park_array_data(PyTuple_GetItem(args, 0), "observations", NPY_FLOAT32, observation_count);
    env->actions = park_array_data(PyTuple_GetItem(args, 1), "actions", NPY_INT32, agent_count);
    env->rewards = park_array_data(PyTuple_GetItem(args, 2), "rewards", NPY_FLOAT32, agent_count);
    env->terminals = park_array_data(PyTuple_GetItem(args, 3), "terminals", NPY_BOOL, agent_count);
    env->truncations = park_array_data(PyTuple_GetItem(args, 4), "truncations", NPY_BOOL, agent_count);
    env->masks = park_array_data(PyTuple_GetItem(args, 5), "masks", NPY_BOOL, agent_count);
    int bound = env->observations && env->actions && env->rewards && env->terminals && env->truncations && env->masks;
    return bound ? 0 : -1;
}

static int park_unpack_seed(PyObject *object, uint64_t *seed_out) {
    long long seed = PyLong_AsLongLong(object);
    if (PyErr_Occurred()) {
        return -1;
    }
    if (seed < 0) {
        PyErr_SetString(PyExc_ValueError, "seed must be a non-negative 63-bit integer");
        return -1;
    }
    *seed_out = (uint64_t) seed;
    return 0;
}

static void park_destroy_env(Park *env) {
    park_close(env);
    if (env->bank != NULL) {
        park_scene_bank_release(env->bank);
    }
    free(env);
}

static PyObject *env_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    if (PyTuple_Size(args) != PARK_ENV_INIT_POSITIONAL_ARGS || kwargs == NULL) {
        PyErr_SetString(PyExc_TypeError, "env_init requires 7 positional arguments and keyword configuration");
        return NULL;
    }
    ParkConfig config;
    memset(&config, 0, sizeof(config));
    uint64_t seed;
    if (park_parse_config(kwargs, &config) != 0 || park_unpack_seed(PyTuple_GetItem(args, 6), &seed) != 0) {
        return NULL;
    }
    PyObject *path_object = PyDict_GetItemString(kwargs, "scene_bank_path");
    const char *path = path_object != NULL ? PyUnicode_AsUTF8(path_object) : NULL;
    if (path == NULL) {
        PyErr_SetString(PyExc_TypeError, "scene_bank_path must be a string");
        return NULL;
    }
    char error[PARK_ERROR_MESSAGE_CAPACITY];
    ParkSceneBank *bank = park_scene_bank_acquire(path, error, sizeof(error));
    if (bank == NULL) {
        PyErr_SetString(PyExc_ValueError, error);
        return NULL;
    }
    Park *env = (Park *) calloc(1, sizeof(Park));
    if (env == NULL) {
        park_scene_bank_release(bank);
        return PyErr_NoMemory();
    }
    if (park_init(env, &config, bank, seed, error, sizeof(error)) != PARK_ERROR_OK) {
        env->bank = bank;
        park_destroy_env(env);
        PyErr_SetString(PyExc_ValueError, error);
        return NULL;
    }
    if (park_bind_buffers(env, args) != 0) {
        park_destroy_env(env);
        return NULL;
    }
    return PyLong_FromVoidPtr(env);
}

static Park *park_unpack_env(PyObject *handle) {
    Park *env = (Park *) PyLong_AsVoidPtr(handle);
    if (env == NULL && !PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "invalid env handle");
    }
    return env;
}

static ParkVecEnv *park_unpack_vec(PyObject *args) {
    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError, "missing vec env handle");
        return NULL;
    }
    ParkVecEnv *vec = (ParkVecEnv *) PyLong_AsVoidPtr(PyTuple_GetItem(args, 0));
    if (vec == NULL || vec->num_envs <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid vec env handle");
        return NULL;
    }
    return vec;
}

static PyObject *env_close(PyObject *self, PyObject *args) {
    Park *env = park_unpack_env(PyTuple_GetItem(args, 0));
    if (env == NULL) {
        return NULL;
    }
    park_destroy_env(env);
    Py_RETURN_NONE;
}

static PyObject *vectorize(PyObject *self, PyObject *args) {
    int num_envs = (int) PyTuple_Size(args);
    if (num_envs <= 0) {
        PyErr_SetString(PyExc_TypeError, "vectorize requires at least one env handle");
        return NULL;
    }
    ParkVecEnv *vec = (ParkVecEnv *) calloc(1, sizeof(ParkVecEnv));
    Park **envs = (Park **) calloc((size_t) num_envs, sizeof(Park *));
    if (vec == NULL || envs == NULL) {
        free(vec);
        free(envs);
        return PyErr_NoMemory();
    }
    vec->envs = envs;
    vec->num_envs = num_envs;
    for (int env_idx = 0; env_idx < num_envs; env_idx++) {
        vec->envs[env_idx] = park_unpack_env(PyTuple_GetItem(args, env_idx));
        if (vec->envs[env_idx] == NULL) {
            free(envs);
            free(vec);
            return NULL;
        }
    }
    return PyLong_FromVoidPtr(vec);
}

static PyObject *vec_reset(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL) {
        return NULL;
    }
    PyObject *seeds = PyTuple_Size(args) > 1 ? PyTuple_GetItem(args, 1) : Py_None;
    if (seeds != Py_None && (!PyList_Check(seeds) || PyList_Size(seeds) != vec->num_envs)) {
        PyErr_Format(PyExc_ValueError, "vec_reset seeds must be a list with one entry per env (%d)", vec->num_envs);
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs && seeds != Py_None; env_idx++) {
        uint64_t seed;
        if (park_unpack_seed(PyList_GetItem(seeds, env_idx), &seed) != 0) {
            return NULL;
        }
        park_rng_seed(&vec->envs[env_idx]->rng, seed);
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        c_reset(vec->envs[env_idx]);
    }
    Py_RETURN_NONE;
}

static PyObject *vec_step(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        c_step(vec->envs[env_idx]);
    }
    Py_RETURN_NONE;
}

static int park_dict_set_float(PyObject *dict, const char *key, float value) {
    PyObject *number = PyFloat_FromDouble((double) value);
    if (number == NULL) {
        return -1;
    }
    int status = PyDict_SetItemString(dict, key, number);
    Py_DECREF(number);
    return status;
}

static PyObject *park_log_to_dict(const ParkLog *log) {
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }
    int status = 0;
    status |= park_dict_set_float(dict, "n", log->n);
    status |= park_dict_set_float(dict, "score", log->score);
    status |= park_dict_set_float(dict, "success_rate", log->success_rate);
    status |= park_dict_set_float(dict, "collision_rate", log->collision_rate);
    status |= park_dict_set_float(dict, "collision_vehicle_rate", log->collision_vehicle_rate);
    status |= park_dict_set_float(dict, "collision_static_rate", log->collision_static_rate);
    status |= park_dict_set_float(dict, "offroad_rate", log->offroad_rate);
    status |= park_dict_set_float(dict, "refine_timeout_rate", log->refine_timeout_rate);
    status |= park_dict_set_float(dict, "horizon_timeout_rate", log->horizon_timeout_rate);
    status |= park_dict_set_float(dict, "handoff_rate", log->handoff_rate);
    status |= park_dict_set_float(dict, "refine_entered_rate", log->refine_entered_rate);
    status |= park_dict_set_float(dict, "path_length_m", log->path_length_m);
    status |= park_dict_set_float(dict, "maneuver_count", log->maneuver_count);
    status |= park_dict_set_float(dict, "episode_return", log->episode_return);
    status |= park_dict_set_float(dict, "episode_length", log->episode_length);
    status |= park_dict_set_float(dict, "mean_threat", log->mean_threat);
    status |= park_dict_set_float(dict, "imitation_reward", log->imitation_reward);
    status |= park_dict_set_float(dict, "scene_agent_count", log->scene_agent_count);
    status |= park_dict_set_float(dict, "scene_tier", log->scene_tier);
    // Terminal-precision metrics average over successful episodes only.
    if (log->success_rate > 0.0f) {
        status
            |= park_dict_set_float(dict, "success_position_error_m", log->success_position_error_m / log->success_rate);
        status |= park_dict_set_float(
            dict,
            "success_heading_error_deg",
            log->success_heading_error_deg / log->success_rate);
    }
    if (status != 0) {
        Py_DECREF(dict);
        return NULL;
    }
    return dict;
}

static PyObject *vec_log(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL) {
        return NULL;
    }
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_log requires the vec handle and a minimum episode count");
        return NULL;
    }
    double min_episodes = PyFloat_AsDouble(PyTuple_GetItem(args, 1));
    if (PyErr_Occurred()) {
        return NULL;
    }
    const int field_count = (int) (sizeof(ParkLog) / sizeof(float));
    ParkLog aggregate;
    memset(&aggregate, 0, sizeof(aggregate));
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        const float *fields = (const float *) &vec->envs[env_idx]->log;
        for (int field_idx = 0; field_idx < field_count; field_idx++) {
            ((float *) &aggregate)[field_idx] += fields[field_idx];
        }
    }
    if (aggregate.n <= 0.0f || aggregate.n < (float) min_episodes) {
        return PyDict_New();
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        memset(&vec->envs[env_idx]->log, 0, sizeof(ParkLog));
    }
    float episode_count = aggregate.n;
    for (int field_idx = 0; field_idx < field_count; field_idx++) {
        ((float *) &aggregate)[field_idx] /= episode_count;
    }
    aggregate.n = episode_count;
    return park_log_to_dict(&aggregate);
}

static PyObject *vec_close(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        park_destroy_env(vec->envs[env_idx]);
    }
    free(vec->envs);
    free(vec);
    Py_RETURN_NONE;
}

static PyObject *vec_teacher_actions(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL || PyTuple_Size(args) != 2) {
        return vec == NULL ? NULL : PyErr_Format(PyExc_TypeError, "vec_teacher_actions requires an output array");
    }
    int agents_per_env = vec->envs[0]->config.max_agents;
    int32_t *actions = park_array_data(
        PyTuple_GetItem(args, 1),
        "teacher actions",
        NPY_INT32,
        (npy_intp) vec->num_envs * agents_per_env);
    if (actions == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        for (int agent_idx = 0; agent_idx < agents_per_env; agent_idx++) {
            actions[env_idx * agents_per_env + agent_idx] = park_teacher_action(vec->envs[env_idx], agent_idx);
        }
    }
    Py_RETURN_NONE;
}

static void park_write_agent_state(const ParkAgent *agent, float *state) {
    const float values[PARK_AGENT_STATE_FIELDS] = {
        agent->pose.x,
        agent->pose.y,
        agent->pose.heading_rad,
        agent->speed_mps,
        agent->steer_rad,
        (float) agent->status,
        (float) agent->phase,
        (float) agent->outcome,
        agent->goal.x,
        agent->goal.y,
        agent->goal.heading_rad,
        agent->threat_intensity,
    };
    memcpy(state, values, sizeof(values));
}

static PyObject *vec_agent_states(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL || PyTuple_Size(args) != 2) {
        return vec == NULL ? NULL : PyErr_Format(PyExc_TypeError, "vec_agent_states requires an output array");
    }
    int agents_per_env = vec->envs[0]->config.max_agents;
    float *states = park_array_data(
        PyTuple_GetItem(args, 1),
        "agent states",
        NPY_FLOAT32,
        (npy_intp) vec->num_envs * agents_per_env * PARK_AGENT_STATE_FIELDS);
    if (states == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        for (int agent_idx = 0; agent_idx < agents_per_env; agent_idx++) {
            int row = env_idx * agents_per_env + agent_idx;
            park_write_agent_state(&vec->envs[env_idx]->agents[agent_idx], states + row * PARK_AGENT_STATE_FIELDS);
        }
    }
    Py_RETURN_NONE;
}

static PyObject *vec_scene_indices(PyObject *self, PyObject *args) {
    ParkVecEnv *vec = park_unpack_vec(args);
    if (vec == NULL) {
        return NULL;
    }
    PyObject *scene_indices = PyList_New(vec->num_envs);
    if (scene_indices == NULL) {
        return NULL;
    }
    for (int env_idx = 0; env_idx < vec->num_envs; env_idx++) {
        PyList_SET_ITEM(scene_indices, env_idx, PyLong_FromLong(vec->envs[env_idx]->scene_idx));
    }
    return scene_indices;
}

static PyObject *scene_bank_info(PyObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }
    char error[PARK_ERROR_MESSAGE_CAPACITY];
    ParkSceneBank *bank = park_scene_bank_acquire(path, error, sizeof(error));
    if (bank == NULL) {
        PyErr_SetString(PyExc_ValueError, error);
        return NULL;
    }
    PyObject *info = Py_BuildValue(
        "{s:i,s:i,s:i,s:f,s:f,s:f,s:(iii)}",
        "lot_count",
        bank->lot_count,
        "scene_count",
        bank->scene_count,
        "max_scene_agent_count",
        bank->max_scene_agent_count,
        "vehicle_length_m",
        (double) bank->vehicle_length_m,
        "vehicle_width_m",
        (double) bank->vehicle_width_m,
        "wheelbase_m",
        (double) bank->wheelbase_m,
        "tier_scene_counts",
        bank->tier_scene_offsets[1] - bank->tier_scene_offsets[0],
        bank->tier_scene_offsets[2] - bank->tier_scene_offsets[1],
        bank->tier_scene_offsets[3] - bank->tier_scene_offsets[2]);
    park_scene_bank_release(bank);
    return info;
}

static int park_parse_pose(PyObject *object, ParkPose *pose) {
    return PyArg_ParseTuple(object, "fff", &pose->x, &pose->y, &pose->heading_rad) ? 0 : -1;
}

static PyObject *plan_maneuver(PyObject *self, PyObject *args) {
    PyObject *edges_object, *bounds_object, *start_object, *goal_object;
    float vehicle_length_m, vehicle_width_m, wheelbase_m;
    if (!PyArg_ParseTuple(
            args,
            "OOOOfff",
            &edges_object,
            &bounds_object,
            &start_object,
            &goal_object,
            &vehicle_length_m,
            &vehicle_width_m,
            &wheelbase_m)) {
        return NULL;
    }
    if (!PyArray_Check(edges_object) || PyArray_NDIM((PyArrayObject *) edges_object) != 2
        || PyArray_DIM((PyArrayObject *) edges_object, 1) != PARK_SEGMENT_FIELDS) {
        PyErr_SetString(PyExc_ValueError, "edges must be a float32 array of shape (count, 4)");
        return NULL;
    }
    npy_intp edge_count = PyArray_DIM((PyArrayObject *) edges_object, 0);
    const ParkSegment *edges = park_array_data(edges_object, "edges", NPY_FLOAT32, edge_count * PARK_SEGMENT_FIELDS);
    ParkBounds bounds;
    ParkPose start, goal;
    if (edges == NULL
        || !PyArg_ParseTuple(bounds_object, "ffff", &bounds.min_x, &bounds.min_y, &bounds.max_x, &bounds.max_y)
        || park_parse_pose(start_object, &start) != 0 || park_parse_pose(goal_object, &goal) != 0) {
        return NULL;
    }
    ParkPlanPoint *points = (ParkPlanPoint *) calloc(PARK_MAX_PLAN_POINTS_PER_AGENT, sizeof(ParkPlanPoint));
    if (points == NULL) {
        return PyErr_NoMemory();
    }
    ParkPlanOutput output = {points, 0, PARK_MAX_PLAN_POINTS_PER_AGENT};
    float cost = 0.0f;
    int status = PARK_PLAN_NOT_FOUND;
    Py_BEGIN_ALLOW_THREADS;
    status = park_plan_maneuver(
        edges,
        (int) edge_count,
        bounds,
        start,
        goal,
        vehicle_length_m,
        vehicle_width_m,
        wheelbase_m,
        &output,
        &cost);
    Py_END_ALLOW_THREADS;
    if (status != PARK_PLAN_OK) {
        free(points);
        return Py_BuildValue("(OOi)", Py_None, Py_None, status);
    }
    npy_intp dims[2] = {output.count, PARK_PLAN_FIELDS};
    PyObject *plan = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (plan == NULL) {
        free(points);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *) plan), points, (size_t) output.count * sizeof(ParkPlanPoint));
    free(points);
    PyObject *result = Py_BuildValue("(Ndi)", plan, (double) cost, status);
    return result;
}

static int park_add_float_tuple(PyObject *module, const char *name, const float *values, int value_count) {
    PyObject *tuple = PyTuple_New(value_count);
    if (tuple == NULL) {
        return -1;
    }
    for (int value_idx = 0; value_idx < value_count; value_idx++) {
        PyTuple_SET_ITEM(tuple, value_idx, PyFloat_FromDouble((double) values[value_idx]));
    }
    if (PyModule_AddObject(module, name, tuple) < 0) {
        Py_DECREF(tuple);
        return -1;
    }
    return 0;
}

static PyMethodDef park_methods[] = {
    {"env_init",
     (PyCFunction) (void (*)(void)) env_init,
     METH_VARARGS | METH_KEYWORDS,
     "Create one parking env over shared buffers"},
    {"env_close", env_close, METH_VARARGS, "Close one parking env"},
    {"vectorize", vectorize, METH_VARARGS, "Group env handles into a vector"},
    {"vec_reset", vec_reset, METH_VARARGS, "Reset every env, optionally reseeding"},
    {"vec_step", vec_step, METH_VARARGS, "Step every env"},
    {"vec_log", vec_log, METH_VARARGS, "Average completed agent-episode metrics"},
    {"vec_close", vec_close, METH_VARARGS, "Close every env and the vector"},
    {"vec_teacher_actions", vec_teacher_actions, METH_VARARGS, "Scripted route + Stanley actions per agent"},
    {"vec_agent_states", vec_agent_states, METH_VARARGS, "Agent poses, phases and outcomes"},
    {"vec_scene_indices", vec_scene_indices, METH_VARARGS, "Current scene index per env"},
    {"scene_bank_info", scene_bank_info, METH_VARARGS, "Validate a scene bank and report its sizes"},
    {"plan_maneuver", plan_maneuver, METH_VARARGS, "Hybrid A* + Reeds-Shepp maneuver between two center poses"},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef park_module = {PyModuleDef_HEAD_INIT, .m_name = "binding", .m_size = -1, .m_methods = park_methods};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();
    PyObject *module = PyModule_Create(&park_module);
    if (module == NULL) {
        return NULL;
    }
    int status = 0;
    status |= PyModule_AddIntConstant(module, "NUM_ACTIONS", PARK_NUM_ACTIONS);
    status |= PyModule_AddIntConstant(module, "NUM_ACCEL_ACTIONS", PARK_NUM_ACCEL_ACTIONS);
    status |= PyModule_AddIntConstant(module, "NUM_STEER_ACTIONS", PARK_NUM_STEER_ACTIONS);
    status |= PyModule_AddIntConstant(module, "EGO_FEATURES", PARK_EGO_FEATURES);
    status |= PyModule_AddIntConstant(module, "EGO_PHASE_OFFSET", PARK_EGO_PHASE_OFFSET);
    status |= PyModule_AddIntConstant(module, "CONDITIONING_FEATURES", PARK_CONDITIONING_FEATURES);
    status |= PyModule_AddIntConstant(module, "TAIL_FEATURES", PARK_TAIL_FEATURES);
    status |= PyModule_AddIntConstant(module, "TAIL_THREAT_IDX", PARK_TAIL_THREAT_IDX);
    status |= PyModule_AddIntConstant(module, "TAIL_TEACHER_ACCEL_IDX", PARK_TAIL_TEACHER_ACCEL_IDX);
    status |= PyModule_AddIntConstant(module, "TAIL_TEACHER_STEER_IDX", PARK_TAIL_TEACHER_STEER_IDX);
    status |= PyModule_AddIntConstant(module, "PARTNER_FEATURES", PARK_PARTNER_FEATURES);
    status |= PyModule_AddIntConstant(module, "LANE_FEATURES", PARK_LANE_FEATURES);
    status |= PyModule_AddIntConstant(module, "BOUNDARY_FEATURES", PARK_BOUNDARY_FEATURES);
    status |= PyModule_AddIntConstant(module, "COUNT_FEATURES", PARK_COUNT_FEATURES);
    status |= PyModule_AddIntConstant(module, "NUM_PHASES", PARK_NUM_PHASES);
    status |= PyModule_AddIntConstant(module, "PHASE_NAVIGATE", PARK_PHASE_NAVIGATE);
    status |= PyModule_AddIntConstant(module, "PHASE_MANEUVER", PARK_PHASE_MANEUVER);
    status |= PyModule_AddIntConstant(module, "PHASE_REFINE", PARK_PHASE_REFINE);
    status |= PyModule_AddIntConstant(module, "STATUS_ACTIVE", PARK_STATUS_ACTIVE);
    status |= PyModule_AddIntConstant(module, "STATUS_PARKED", PARK_STATUS_PARKED);
    status |= PyModule_AddIntConstant(module, "STATUS_REMOVED", PARK_STATUS_REMOVED);
    status |= PyModule_AddIntConstant(module, "OUTCOME_SUCCESS", PARK_OUTCOME_SUCCESS);
    status |= PyModule_AddIntConstant(module, "AGENT_STATE_FIELDS", PARK_AGENT_STATE_FIELDS);
    status |= PyModule_AddIntConstant(module, "MAX_AGENTS_PER_SCENE", PARK_MAX_AGENTS_PER_SCENE);
    status |= PyModule_AddIntConstant(module, "MAX_PARTNER_SLOTS", PARK_MAX_PARTNER_SLOTS);
    status |= PyModule_AddIntConstant(module, "MAX_LANE_SLOTS", PARK_MAX_LANE_SLOTS);
    status |= PyModule_AddIntConstant(module, "MAX_BOUNDARY_SLOTS", PARK_MAX_BOUNDARY_SLOTS);
    status |= PyModule_AddIntConstant(module, "SCENE_BANK_VERSION", PARK_SCENE_BANK_VERSION);
    status |= PyModule_AddIntConstant(module, "PLAN_OK", PARK_PLAN_OK);
    status |= PyModule_AddIntConstant(module, "EDGE_SOURCE_CURB", PARK_EDGE_SOURCE_CURB);
    status |= PyModule_AddIntConstant(module, "EDGE_SOURCE_OBSTACLE", PARK_EDGE_SOURCE_OBSTACLE);
    status |= PyModule_AddIntConstant(module, "EDGE_SOURCE_STATIC_VEHICLE", PARK_EDGE_SOURCE_STATIC_VEHICLE);
    status |= PyModule_AddObject(module, "MAX_ACCEL_MPS2", PyFloat_FromDouble(PARK_MAX_ACCEL_MPS2));
    status |= PyModule_AddObject(module, "MAX_STEER_RAD", PyFloat_FromDouble(PARK_MAX_STEER_RAD));
    status |= PyModule_AddObject(module, "PLANNER_MAX_STEER_RAD", PyFloat_FromDouble(PARK_PLANNER_MAX_STEER_RAD));
    status |= park_add_float_tuple(module, "ACCEL_VALUES_MPS2", PARK_ACCEL_VALUES_MPS2, PARK_NUM_ACCEL_ACTIONS);
    status |= park_add_float_tuple(module, "STEER_VALUES_RAD", PARK_STEER_VALUES_RAD, PARK_NUM_STEER_ACTIONS);
    if (status != 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
