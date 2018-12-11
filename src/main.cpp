// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Dense>

// Timer
#include <chrono>

#include <iostream>
#include <string>
#include <math.h>
#define PI 3.14159265
#define NUM_POINTS 4

#define REAL double
#define VOID int

extern "C"
{
#include "triangle.h"
}

extern "C" void triangulate(char *, struct triangulateio *, struct triangulateio *,struct triangulateio *);


// the level of detail for the curve
int LOD=20;

using namespace std;
using namespace Eigen;

// VertexBufferObject wrapper
VertexBufferObject VBO;

// Contains the vertex positions
Eigen::MatrixXf V(2,3);
Eigen::MatrixXi E(3,2);
Eigen::MatrixXf P_tr(2,3*NUM_POINTS);
Eigen::MatrixXf P_sq(2,4*NUM_POINTS);
Eigen::MatrixXf P_hex(2,6*NUM_POINTS);

// Contains the view transformation
Eigen::Matrix4f view(4,4);

// globals
/*
int icc = 0;
bool insMode = false;
int numT = 0;
bool transMode = false;
int transIndex = -1;
double prevX;
double prevY;
bool delMode = false;
bool transP = false;
MatrixXf colors(3,9);
bool colorMode = false;
int colorIndex = -1;
Matrix3f transColor;
bool animMode = false;
int animIndex = -1;
auto t_start = std::chrono::high_resolution_clock::now();;
double animX;
double animY;
bool animate = false;
*/
bool trMode = false;
bool sqMode = false;
bool hexMode = false;
int pIndex = -1;

 
/*
template <
 typename DerivedV,
 typename DerivedE,
 typename DerivedH,
 typename DerivedVM,
 typename DerivedEM,
 typename DerivedV2,
 typename DerivedF2,
 typename DerivedVM2,
 typename DerivedEM2>
 */
inline void mytriangulate( Eigen::MatrixXf & V, Eigen::MatrixXi & E, Eigen::MatrixXf & H,
  Eigen::VectorXi & VM, Eigen::VectorXi & EM, std::string flags,
  Eigen::MatrixXf & V2, Eigen::MatrixXi & F2,
  Eigen::VectorXi & VM2, Eigen::VectorXi & EM2)
{

  assert( (VM.size() == 0 || V.rows() == VM.size()) && 
    "Vertex markers must be empty or same size as V");
  assert( (EM.size() == 0 || E.rows() == EM.size()) && 
    "Segment markers must be empty or same size as E");
  assert(V.cols() == 2);
  assert(E.size() == 0 || E.cols() == 2);
  assert(H.size() == 0 || H.cols() == 2);

  // Prepare the flags
  string full_flags = flags + "pz" + (EM.size() || VM.size() ? "" : "B");

  typedef Map< Matrix<double,Dynamic,Dynamic,RowMajor> > MapXdr;
  typedef Map< Matrix<int,Dynamic,Dynamic,RowMajor> > MapXir;

  // Prepare the input struct
  triangulateio in;
  in.numberofpoints = V.rows();
  in.pointlist = (double*)calloc(V.size(),sizeof(double));
  {
    MapXdr inpl(in.pointlist,V.rows(),V.cols());
    inpl = V.cast<double>();
  }

  in.numberofpointattributes = 0;
  in.pointmarkerlist = (int*)calloc(V.size(),sizeof(int)) ;
  for(unsigned i=0;i<V.rows();++i) in.pointmarkerlist[i] = VM.size()?VM(i):1;

  in.trianglelist = NULL;
  in.numberoftriangles = 0;
  in.numberofcorners = 0;
  in.numberoftriangleattributes = 0;
  in.triangleattributelist = NULL;

  in.numberofsegments = E.size()?E.rows():0;
  in.segmentlist = (int*)calloc(E.size(),sizeof(int));
  {
    MapXir insl(in.segmentlist,E.rows(),E.cols());
    insl = E.cast<int>();
  }
  in.segmentmarkerlist = (int*)calloc(E.rows(),sizeof(int));
  for (unsigned i=0;i<E.rows();++i) in.segmentmarkerlist[i] = EM.size()?EM(i):1;

  in.numberofholes = H.size()?H.rows():0;
  in.holelist = (double*)calloc(H.size(),sizeof(double));
  {
    MapXdr inhl(in.holelist,H.rows(),H.cols());
    inhl = H.cast<double>();
  }
  in.numberofregions = 0;

  // Prepare the output struct
  triangulateio out;
  out.pointlist = NULL;
  out.trianglelist = NULL;
  out.segmentlist = NULL;
  out.segmentmarkerlist = NULL;
  out.pointmarkerlist = NULL;

  // Call triangle
  triangulate(const_cast<char*>(full_flags.c_str()), &in, &out, 0);

  // Return the mesh
  V2 = MapXdr(out.pointlist,out.numberofpoints,2).cast<float>();
  F2 = MapXir(out.trianglelist,out.numberoftriangles,3).cast<int>();
  if(VM.size())
  {
    VM2 = MapXir(out.pointmarkerlist,out.numberofpoints,1).cast<int>();
  }
  if(EM.size())
  {
    EM2 = MapXir(out.segmentmarkerlist,out.numberofsegments,1).cast<int>();
  }

  // Cleanup in
  free(in.pointlist);
  free(in.pointmarkerlist);
  free(in.segmentlist);
  free(in.segmentmarkerlist);
  free(in.holelist);
  // Cleanup out
  free(out.pointlist);
  free(out.trianglelist);
  free(out.segmentlist);
  free(out.segmentmarkerlist);
  free(out.pointmarkerlist);
}

/*
template <
 typename DerivedV,
 typename DerivedE,
 typename DerivedH,
 typename DerivedV2,
 typename DerivedF2>
 */
inline void mytriangulate(Eigen::MatrixXf & V, Eigen::MatrixXi & E, Eigen::MatrixXf & H, std::string flags,
  Eigen::MatrixXf & V2,
  Eigen::MatrixXi & F2)
{
  Eigen::VectorXi VM,EM,VM2,EM2;
  return mytriangulate(V,E,H,VM,EM,flags,V2,F2,VM2,EM2);
}

//template void triangulate<Eigen::Matrix<double, -1, -1, 1, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 1, -1, -1>, Eigen::Matrix<double, -1, -1, 1, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> > const&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 1, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);
// generated by autoexplicit.sh
//template void triangulate<Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<double, -1, -1, 0, -1, -1>, Eigen::Matrix<int, -1, -1, 0, -1, -1> >(Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, std::basic_string<char, std::char_traits<char>, std::allocator<char> >, Eigen::PlainObjectBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> >&, Eigen::PlainObjectBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> >&);


void fillEdges() {
	int tmp = LOD*(NUM_POINTS+1);
	E.resize(tmp*3,2);
	int count = 0;
	for (int i=0; i<3; i++) {
		for (int j=0; j<tmp; j++) {
			E.row(count) << count+i, count+1+i;//i*tmp+i+j,i*tmp+j+1;
			count++;
		}
	}
}


Vector2f GetPoint(int i, int ubound, int lbound) {
	// return 1st point
	if (i<0)
		return P_tr.col(lbound);
	// return last point
	if (i<NUM_POINTS)
		return P_tr.col(lbound+i);

	return P_tr.col(ubound-1);
}

void tessellate_triangle() {
	// P_tr contains the control points
	trMode = true;
	V.resize(2,LOD*(NUM_POINTS+1)*3+3+P_tr.cols()-6); //CHANGED
	MatrixXf Vin(2,LOD*(NUM_POINTS+1)*3+3);
	Vin = V.block(0,0,2,LOD*(NUM_POINTS+1)*3+3);
	Vin.transposeInPlace();
	MatrixXf Vout;
	MatrixXi Eout;
	MatrixXf H;

	mytriangulate(Vin,E,H,"",Vout,Eout);


	IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	cout << "P_tr: " << P_tr.format(OctaveFmt) << endl;

	int count = 0;
	for (int seg = 0; seg<3; seg++) {
		//V.col(count) << 0.,0.;
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		//count++;
		for(int start_cv=-3,j=0;j!=NUM_POINTS+1;++j,++start_cv)
		{
				// for each section of curve, draw LOD number of divisions
			for(int i=0;i!=LOD;++i) {

				// use the parametric time value 0 to 1 for this curve
				// segment.
				float t = (float)i/LOD;

				// the t value inverted
				float it = 1.0f-t;

				// calculate blending functions for cubic bspline
				float b0 = it*it*it/6.0f;
				float b1 = (3*t*t*t - 6*t*t +4)/6.0f;
				float b2 = (-3*t*t*t +3*t*t + 3*t + 1)/6.0f;
				float b3 =  t*t*t/6.0f;

				// calculate the x,y and z of the curve point
				float x = b0 * GetPoint( start_cv + 0 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b1 * GetPoint( start_cv + 1 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b2 * GetPoint( start_cv + 2 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b3 * GetPoint( start_cv + 3 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) ;

				float y = b0 * GetPoint( start_cv + 0 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b1 * GetPoint( start_cv + 1 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b2 * GetPoint( start_cv + 2 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b3 * GetPoint( start_cv + 3 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) ;

				V.col(count) << x,y;
				cout << "index: " << count << endl;
				cout << "vertex: " << x << ", " << y << endl;
				count++;
			}
		}
		V.col(count) = P_tr.col(NUM_POINTS*(seg+1)-1);
		cout << "index: " << count << endl;
		cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}
	//int numCols = V.cols();
	//V.conservativeResize(2,numCols+P_tr.cols()-6);
	//V.block(0,numCols,2,P_tr.cols()) = P_tr;
	/*
	for (int i=count; i<V.cols(); i++) {
		cout << "index: " << i << endl;
		cout << "vertex: " << V(0,i) << ", " << V(1,i) << endl;
	}
	*/
	for (int i=1; i<P_tr.cols()-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1)
			continue;
		V.col(count) = P_tr.col(i);
		cout << "index: " << count << endl;
		cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}

}

/*
int closestVertex(double xworld, double yworld, int cv) {
    float min_dist = numeric_limits<float>::max();
    float dist;
    for (int i=0; i<V.cols(); i++) {
        dist = sqrt((V(0,i)-xworld)*(V(0,i)-xworld)+(V(1,i)-yworld)*(V(1,i)-yworld));
        if (dist<min_dist) {
            min_dist = dist;
            cv = i;
        }
    }
    return cv;
}

void pan(int width, int height, float x, float y) {
    float xworld = 2*abs(x)-1; //=(x*width/width)*2-1
    float yworld = 2*abs(y)-1; //=(y*height/height)*2-1
    view(0,3) += ((x>0.0)-(x<0.0))*xworld;
    view(1,3) += ((y>0.0)-(y<0.0))*yworld;
}

void zoom(int width, int height, float z) {
    // since we're in (-1,1) no translation required?
    view(0,0) *= (z+1);
    view(1,1) *= (z+1);
}

void setMode(bool insMode0,bool transMode0,int transIndex0,bool delMode0,bool colorMode0,
    int colorIndex0,bool animMode0,int animIndex0) {
    insMode = insMode0;
    transMode = transMode0;
    transIndex = transIndex0;
    delMode = delMode0;
    colorMode = colorMode0;
    colorIndex = colorIndex0;
    animMode = animMode0;
    animIndex = animIndex0;
}

void setColor(int key) {
    if (colorMode && colorIndex != -1)
        C.col(colorIndex) = colors.col(key-1);
} 

Vector2f barycenter() {
    Vector2f c((V(0,transIndex)+V(0,transIndex+1)+V(0,transIndex+2))/3,
        (V(1,transIndex)+V(1,transIndex+1)+V(1,transIndex+2))/3);
    return c;
}

void rotate(float deg) {
    Vector2f c = barycenter();
    deg = -deg*(PI/180);
    Matrix2f rot;
    rot << cos(deg), -sin(deg), sin(deg), cos(deg);
    for (int i=transIndex; i<(transIndex+3); i++) {
        V.col(i) -= c;
        V.col(i) = rot*V.col(i);
        V.col(i) += c;
    }
}

void scale(float sc) {
    Vector2f c = barycenter();
    sc += 1;
    Matrix2f scl;
    scl << sc, 0, 0, sc;
    for (int i=transIndex; i<(transIndex+3); i++) {
        V.col(i) -= c;
        V.col(i) = scl*V.col(i);
        V.col(i) += c;
    }
}

pair<MatrixXf,MatrixXf> removeColumns(int col) {
    MatrixXf newV(2,V.cols()-3);
    MatrixXf newC(3,C.cols()-3);
    int j = 0;
    for (int i=0; i<V.cols(); i++) {
        if (i!=col && i!=(col+1) && i!=(col+2)) {
            newV.col(j) = V.col(i);
            newC.col(j) = C.col(i);
            j += 1;
        }
    }
    pair<MatrixXf,MatrixXf> p(newV,newC);
    return p;
}

int triangle_click(double xworld, double yworld) {
    // all in world coordinates (-1,1)
    Matrix3f A;
    Vector3f b;
    Vector3f tmp;
    Vector3f x;
    float detA;

    for (int i=0; i<V.cols(); i+=3) {
        A << V(0,i),V(0,i+1),V(0,i+2),V(1,i),V(1,i+1),V(1,i+2),1,1,1;
        b << xworld, yworld, 1;
        detA = A.determinant();
        if (detA == 0)
            continue;
        for (int j=0; j<3; j++) {
            tmp = A.col(j);
            A.col(j) = b;
            x(j) = A.determinant()/detA;
            A.col(j) = tmp;
        }
        if (x(0)>=0 && x(1)>=0 && x(2)>=0) {
            cout << "triangle clicked" << endl;
            return i;
        }
    }
    
    return -1;
}
*/

int point_tr_click(double xworld, double yworld) {
	float radius = 0.05;
	for (int i=0; i<P_tr.cols(); i++) {
		if (abs(xworld-P_tr(0,i)) < radius && abs(yworld-P_tr(1,i)) < radius) {
			return i;
		}
	}
	return -1;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// callback for glfwSetCursorPos
void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Convert screen position to world coordinates
    Vector4f p_screen(xpos,height-1-ypos,0,1); // NOTE: y axis is flipped in glfw
    Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1);
    Vector4f p_world = view.inverse()*p_canonical;

    double xworld = p_world(0);
    double yworld = p_world(1);

    /*
    if (insMode) {
		if (icc == 1) {
			V.col((numT*3)+1) << xworld, yworld;
			V.col((numT*3)+2) << xworld, yworld;
		} else if (icc == 2) {
			V.col((numT*3)+2) << xworld, yworld;
		}
	} else if (transMode && transP && transIndex != -1) {
        V(0,transIndex) += (xworld-prevX);
        V(0,transIndex+1) += (xworld-prevX);
        V(0,transIndex+2) += (xworld-prevX);
        V(1,transIndex) += (yworld-prevY);
        V(1,transIndex+1) += (yworld-prevY);
        V(1,transIndex+2) += (yworld-prevY);
    }

    prevX = xworld;
    prevY = yworld;
    */

	//VBO.update(V);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // Convert screen position to world coordinates
    // why extra -1 only on y? possibly the close(x) window sign
    Vector4f p_screen(xpos,height-1-ypos,0,1); // NOTE: y axis is flipped in glfw
    Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1);
    Vector4f p_world = view.inverse()*p_canonical;

    double xworld = p_world(0);
    double yworld = p_world(1);

    if (trMode) {
    	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
	    	pIndex = point_tr_click(xworld,yworld);
	    	if (pIndex != -1) {
	    		cout << "index clicked: " << pIndex << endl;
	    		cout << "vertex: " << P_tr(0,pIndex) << " , " << P_tr(1,pIndex) << endl;

	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		P_tr.col(pIndex) << xworld, yworld;
	    		tessellate_triangle();
	    	}

	    }
    }

    /*
    if (insMode && button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    	std::cout << "clicked inserstion mode" << std::endl;
    	if (icc == 0) {
    		V.conservativeResize(2,(numT*3)+3);
            C.conservativeResize(3,(numT*3)+3);
    		V.col((numT*3)) << xworld, yworld;
    		V.col((numT*3)+1) << xworld, yworld;
    		V.col((numT*3)+2) << xworld, yworld;
            C.col((numT*3)) << 0.0,0.0,0.0;
            C.col((numT*3)+1) << 0.0,0.0,0.0;
            C.col((numT*3)+2) << 0.0,0.0,0.0;
    	} else if (icc == 1) {
    		V.col((numT*3)+1) << xworld, yworld;
    		V.col((numT*3)+2) << xworld, yworld;
    	} else if (icc == 2) {
    		V.col((numT*3)+2) << xworld, yworld;
    		C.col((numT*3)) << 1.0,0.0,0.0;
            C.col((numT*3)+1) << 1.0,0.0,0.0;
            C.col((numT*3)+2) << 1.0,0.0,0.0;
    		numT += 1;
    	}
    	icc += 1;
    	icc %= 3;
    } else if (transMode) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            transIndex = triangle_click(xworld, yworld);
            if (transIndex != -1) {
                transP = true;
                transColor.col(0) = C.col(transIndex);
                transColor.col(1) = C.col(transIndex+1);
                transColor.col(2) = C.col(transIndex+2);
                C.col(transIndex) << 0.0,0.0,1.0;
                C.col(transIndex+1) << 0.0,0.0,1.0;
                C.col(transIndex+2) << 0.0,0.0,1.0;
            }
        } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
            transP = false;
            C.col(transIndex) = transColor.col(0);
            C.col(transIndex+1) = transColor.col(1);
            C.col(transIndex+2) = transColor.col(2);
        }
    } else if (delMode && button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        int delIndex = triangle_click(xworld, yworld);
        if (delIndex != -1) {
            pair<MatrixXf,MatrixXf> vc = removeColumns(delIndex);
            V = vc.first;
            C = vc.second;
            numT -= 1;
        } 
    } else if (colorMode && button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        colorIndex = closestVertex(xworld, yworld, colorIndex);
    } else if (animMode) {
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            animIndex = closestVertex(xworld, yworld, animIndex);
        } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
            animX = xworld;
            animY = yworld;
            animate = true;
            t_start = std::chrono::high_resolution_clock::now();
        }
    }
    */

    // Upload the change to the GPU
    VBO.update(V);
    //VBO_C.update(C);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // only on release of key
    if (action == GLFW_RELEASE) {
        //int width, height;
        //glfwGetWindowSize(window, &width, &height);
	    switch (key)
	    {
            case GLFW_KEY_T:
            	tessellate_triangle();
            	break;
	    }
	}

    // Upload the change to the GPU
    VBO.update(V);
}


int main(void)
{
    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, "2D Tesselations", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    #ifndef __APPLE__
      glewExperimental = true;
      GLenum err = glewInit();
      if(GLEW_OK != err)
      {
        /* Problem: glewInit failed, something is seriously wrong. */
       fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
      }
      glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
      fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    #endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

    // Initialize the VAO
    // A Vertex Array Object (or VAO) is an object that describes how the vertex
    // attributes are stored in a Vertex Buffer Object (or VBO). This means that
    // the VAO is not the actual object storing the vertex data,
    // but the descriptor of the vertex data.
    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

    // Initialize the VBO with the vertices data
    // A VBO is a data container that lives in the GPU memory
    VBO.init();
    V.setZero();
    VBO.update(V);

    //control points for triangle
	//P1 << 0.0, 0.5, -0.5, 
	//		0.5, -0.5, -0.5;
    for (int i=0; i<NUM_POINTS; i++) {
		//P2.col(i) << P1(0,i)+(P1(0,(i+1)%3)-P1(0,i))/(NUM_POINTS-1),
		//P1(1,i)+(P1(1,(i+1)%3)-P1(1,i))/(NUM_POINTS-1);
		P_tr.col(i) << 0+i*0.5/(NUM_POINTS-1),0.5-float(i*1)/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		//P3.col(i) << P1(0,i)+2*(P1(0,(i+1)%3)-P1(0,i))/(NUM_POINTS-1),
		//P1(1,i)+2*(P1(1,(i+1)%3)-P1(1,i))/(NUM_POINTS-1);
		P_tr.col(i+NUM_POINTS) << 0-i*0.5/(NUM_POINTS-1),0.5-float(i*1)/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		//P4.col(i) << P1(0,i)+3*(P1(0,(i+1)%3)-P1(0,i))/(NUM_POINTS-1),
		//P1(1,i)+3*(P1(1,(i+1)%3)-P1(1,i))/(NUM_POINTS-1);
		P_tr.col(i+2*NUM_POINTS) << 0.5-float(i*1)/(NUM_POINTS-1),-0.5;
	}

	fillEdges();
    view.setIdentity();

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec2 position;"
                    "uniform mat4 view;"
                    "void main()"
                    "{"
                    "    gl_Position = view*vec4(position,0.0,1.0);"
                    "}";
    const GLchar* fragment_shader =
            "#version 150 core\n"
                    "out vec4 outColor;"
                    "uniform vec3 triangleColor;"
                    "void main()"
                    "{"
                    "    outColor = vec4(triangleColor, 1.0);"
                    "}";

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader,fragment_shader,"outColor");
    program.bind();

    // The vertex shader wants the position of the vertices as an input.
    // The following line connects the VBO we defined above with the position "slot"
    // in the vertex shader
    program.bindVertexAttribArray("position",VBO);

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Update viewport
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // cursor movement callback
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Bind your VAO (not necessary if you have only one)
        VAO.bind();

        // Bind your program
        program.bind();

        glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

	/*
        if (animMode && animIndex != -1 && animate) {
            auto t_now = std::chrono::high_resolution_clock::now();
            float time = chrono::duration_cast<chrono::duration<float>>(t_now-t_start).count();
            if (time <= 1.0) {
                V(0,animIndex) = (1-time)*V(0,animIndex) + time*animX;
                V(1,animIndex) = (1-time)*V(1,animIndex) + time*animY;
            } else {
                animate = false;
            }
            VBO.update(V);
        }
        
    	if (icc == 1) {
    		glDrawArrays(GL_LINES, (numT*3), 2);
    	} else if (icc == 2) {
    		glDrawArrays(GL_LINE_LOOP, (numT*3), 3);
    	}
        
        // Draw triangles
        if (numT > 0) {
        	glDrawArrays(GL_TRIANGLES, 0, (numT*3));
        }
	*/

		if (trMode) {
			//glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.0);

			//glDrawArrays(GL_TRIANGLE_FAN, 0*(NUM_POINTS+1)*LOD+0, (NUM_POINTS+1)*LOD+2);
			//glDrawArrays(GL_TRIANGLE_FAN, 1*(NUM_POINTS+1)*LOD+2, (NUM_POINTS+1)*LOD+2);
			//glDrawArrays(GL_TRIANGLE_FAN, 2*(NUM_POINTS+1)*LOD+4, (NUM_POINTS+1)*LOD+2);

			glUniform3f(program.uniform("triangleColor"),0.5,0.2,0.0);
			glPointSize(5);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS+1)*LOD+3, 3*NUM_POINTS-6);

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			//glLineWidth(3);
			glDrawArrays(GL_LINE_STRIP, 0*(NUM_POINTS+1)*LOD+0, (NUM_POINTS+1)*LOD+1);
			glDrawArrays(GL_LINE_STRIP, 1*(NUM_POINTS+1)*LOD+1, (NUM_POINTS+1)*LOD+1);
			glDrawArrays(GL_LINE_STRIP, 2*(NUM_POINTS+1)*LOD+2, (NUM_POINTS+1)*LOD+1);

		}


        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    VAO.free();
    VBO.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
