#pragma once

#ifdef _DEBUG

#	ifdef _WIN32
#		include <atlbase.h>
#		define ASSERT(expression) _ASSERTE(expression)
#		define ARGUMENT(expression) ASSERT(expression)
#		define VERIFY(expression) ASSERT(expression)
#	endif

#endif

#ifndef ASSERT
#	define ASSERT(expression) if(!expression) throw std::runtime_error(#expression " is invalid");
#endif

#ifndef ARGUMENT
#	define ARGUMENT(expression) if(!expression) throw std::invalid_argument(#expression " is invalid");
#endif

#ifndef VERIFY
#	define VERIFY(expression) if(!expression) throw std::runtime_error(#expression " failed");
#endif

#ifndef PI
#	define PI	3.14159265358979323846
#endif
