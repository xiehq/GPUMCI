#pragma once

#ifdef TEST_SLOW //Run slow tests
#define SLOW_TEST_CASE( ... ) TEST_CASE( __VA_ARGS__ )
#else //Override the test case auto-registration
#define SLOW_TEST_CASE( ... ) static void INTERNAL_CATCH_UNIQUE_NAME(  ____C_A_T_C_H____T_E_S_T____ )() 
#endif
