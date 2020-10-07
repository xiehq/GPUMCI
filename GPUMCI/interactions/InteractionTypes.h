#pragma once

/**
* Interaction types
*/
enum class InteractionType {
	INTERACTION_COMPTON = 0,
	INTERACTION_PHOTO = 1,
	INTERACTION_RAYLEIGH = 3,
	INTERACTION_FICTIOUS = 4
};

/**
* Compton methods
*/
enum class ComptonMethod {
	EVERETT,
	PRECOMPUTED,
	DISABLED
};

/**
* Rayleigh methods
*/
enum class RayleighMethod {
	PRECOMPUTED,
	DISABLED
};
