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
#include <map>

#define PI 3.14159265
#define NUM_POINTS 5
#define LOD 20 // the level of detail for the curve
#define REAL double
#define VOID int

extern "C"
{
#include "triangle.h"
}

extern "C" void triangulate(char *, struct triangulateio *, struct triangulateio *,struct triangulateio *);

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
Eigen::Matrix4f model(4,4);

map<int, int> trMap;
map<int, int> sqMap;
map<int, int> hexMap;

bool trMode = false;
bool lagTrMode = false;
bool sqMode = false;
bool lagSqMode = false;
bool hexMode = false;
bool lagHexMode = false;
int pIndex = -1;
bool finalMode = false;
bool finalLagMode = false;
bool finalsqMode = false;
bool finalLagSqMode = false;
bool finalhexMode = false;
bool finalLagHexMode = false;
int prevCols;
int numCols;
float tr_len;
VectorXf t_ar(NUM_POINTS);
float dx;
float dy;
float xstart;
float ystart;
float xstart2;
float ystart2;

MatrixXf Tr(4,2);
bool pointyTrMode = false;

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

inline void mytriangulate(Eigen::MatrixXf & V, Eigen::MatrixXi & E, Eigen::MatrixXf & H, std::string flags,
  Eigen::MatrixXf & V2,
  Eigen::MatrixXi & F2)
{
  Eigen::VectorXi VM,EM,VM2,EM2;
  return mytriangulate(V,E,H,VM,EM,flags,V2,F2,VM2,EM2);
}


void setMode(bool tr,bool lagtr, bool finaltr, bool finallagtr,
	bool sq,bool lagsq, bool finalsq, bool finallagsq,
	bool hex,bool laghex, bool finalhex, bool finallaghex) {
    trMode = tr;
    lagTrMode = lagtr;
    finalMode = finaltr;
    finalLagMode = finallagtr;
    sqMode = sq;
    lagSqMode = lagsq;
    finalsqMode = finalsq;
    finalLagSqMode = finallagsq;
    hexMode = hex;
    lagHexMode = laghex;
    finalhexMode = finalhex;
    finalLagHexMode = finallaghex;
}

void fillEdges() {
	int rows;
	if (finalMode) {
		rows = LOD*NUM_POINTS*3;
	} else if (finalsqMode) {
		rows = LOD*NUM_POINTS*4;
	} else if (finalhexMode) {
		rows = LOD*NUM_POINTS*6;
	} else if (finalLagMode) {
		rows = LOD*(NUM_POINTS-1)*3;
	} else if (finalLagSqMode) {
		rows = LOD*(NUM_POINTS-1)*4;
	} else if (finalLagHexMode) {
		rows = LOD*(NUM_POINTS-1)*6;
	}
	E.resize(rows,2);
	//cout << "Edges" << endl;
	for (int i=0; i<rows-1; i++) {
		E.row(i) << i, i+1;
		//cout << "row: " << i << ", " << i+1 << endl;
	}
	E.row(rows-1) << rows-1, 0;
	//cout << "row: " << rows-1 << ", " << 0 << endl;
}

Vector2f evalLagrange(float t, int type, int lindex, int uindex) {
	float y1 = 0;
	float y2 = 0;
	float prod1;

	if (type == 0) {
		for (int idash = 0,i=lindex; i<=uindex; i++, idash++) {
			prod1 = 1;
			for (int j=0; j<=4; j++) {
				if (t_ar(idash) == t_ar(j))
					continue;
				prod1 *= (t-t_ar(j))/(t_ar(idash)-t_ar(j));
			}
			y1 += P_tr(0,i)*prod1;
			y2 += P_tr(1,i)*prod1;
		}
	} else if (type == 1) {
		for (int idash = 0,i=lindex; i<=uindex; i++, idash++) {
			prod1 = 1;
			for (int j=0; j<=4; j++) {
				if (t_ar(idash) == t_ar(j))
					continue;
				prod1 *= (t-t_ar(j))/(t_ar(idash)-t_ar(j));
			}
			y1 += P_sq(0,i)*prod1;
			y2 += P_sq(1,i)*prod1;
		}
	} else if (type == 2) {
		for (int idash = 0,i=lindex; i<=uindex; i++, idash++) {
			prod1 = 1;
			for (int j=0; j<=4; j++) {
				if (t_ar(idash) == t_ar(j))
					continue;
				prod1 *= (t-t_ar(j))/(t_ar(idash)-t_ar(j));
			}
			y1 += P_hex(0,i)*prod1;
			y2 += P_hex(1,i)*prod1;
		}
	}

	Vector2f y;
	y << y1, y2;

	return y;
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

Vector2f GetPoint_sq(int i, int ubound, int lbound) {
	// return 1st point
	if (i<0)
		return P_sq.col(lbound);
	// return last point
	if (i<NUM_POINTS)
		return P_sq.col(lbound+i);

	return P_sq.col(ubound-1);
}

Vector2f GetPoint_hex(int i, int ubound, int lbound) {
	// return 1st point
	if (i<0)
		return P_hex.col(lbound);
	// return last point
	if (i<NUM_POINTS)
		return P_hex.col(lbound+i);

	return P_hex.col(ubound-1);
}

MatrixXf prepareVin() {
	int rows;
	if (finalMode) {
		rows = LOD*NUM_POINTS*3;
	} else if (finalsqMode) {
		rows = LOD*NUM_POINTS*4;
	} else if (finalhexMode) {
		rows = LOD*NUM_POINTS*6;
	} else if (finalLagMode) {
		rows = LOD*(NUM_POINTS-1)*3;
	} else if (finalLagSqMode) {
		rows = LOD*(NUM_POINTS-1)*4;
	} else if (finalLagHexMode) {
		rows = LOD*(NUM_POINTS-1)*6;
	}
	MatrixXf Vtmp(2,rows);
	Vtmp = V.block(0,0,2,rows);
	Vtmp.transposeInPlace();
	/*
	// sample fewer points
	//MatrixXf Vin(LOD*3,2);
	//cout << "Vin " << endl;
	for (int i=0; i<Vin.rows(); i++) {
		Vin.row(i) = Vtmp.row(4*i);
		//cout << "index: " << i << endl;
		//cout << "vertex: " << Vin(i,0) << ", " << Vin(i,1) << endl;
	}
	*/
	return Vtmp;
}

void tessellate_triangle() {

	V.resize(2,LOD*NUM_POINTS*3+1+P_tr.cols()-6);

	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//cout << "P_tr: " << P_tr.transpose().format(OctaveFmt) << endl;

	int count = 0;
	for (int seg = 0; seg<3; seg++) {

		for(int start_cv=-2,j=1;j!=NUM_POINTS+1;++j,++start_cv)
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
				//cout << "index: " << count << endl;
				//cout << "vertex: " << x << ", " << y << endl;
				count++;
			}
		}
		if (seg == 2) {
			V.col(count) = P_tr.col(NUM_POINTS*(seg+1)-1);
			//cout << "index: " << count << endl;
			//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
			count++;
		}
	}

	for (int i=1; i<P_tr.cols()-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1)
			continue;
		V.col(count) = P_tr.col(i);
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}
}

void lagrange_triangle() {

	V.resize(2,LOD*(NUM_POINTS-1)*3+1+P_tr.cols()-6); //CHANGED

	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//cout << "P_tr: " << P_tr.transpose().format(OctaveFmt) << endl;

	int count = 0;
	Vector2f y;
	for (int seg = 0; seg<3; seg++) {
		float lindex = seg*NUM_POINTS;
		float uindex = (seg+1)*NUM_POINTS-1;
		int count2 = 0;
		for (int i=0; i<NUM_POINTS-1;i++) {
			for (int j=0; j<LOD;j++) {
				float t = float(count2)*float(1)/float(LOD*(NUM_POINTS-1));
				//cout << "t: " << t << endl;
				y = evalLagrange(t, 0, lindex, uindex);
				V.col(count) = y;
				//cout << "index: " << count << endl;
				//cout << "vertex: " << y(0) << ", " << y(1) << endl;
				count++;
				count2++;
			}
		}

		if (seg == 2) {
			V.col(count) = P_tr.col(0);
			//cout << "index: " << count << endl;
			//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
			count++;
		}
	}

	for (int i=1; i<P_tr.cols()-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1)
			continue;
		V.col(count) = P_tr.col(i);
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}
}

void lagrange_sq() {
	
	V.resize(2,LOD*(NUM_POINTS-1)*4+1+4*NUM_POINTS-8);
	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//cout << "P_sq: " << P_sq.transpose().format(OctaveFmt) << endl;

	int count = 0;
	Vector2f y;
	for (int seg = 0; seg<4; seg++) {
		float lindex = seg*NUM_POINTS;
		float uindex = (seg+1)*NUM_POINTS-1;
		int count2 = 0;
		for (int i=0; i<NUM_POINTS-1;i++) {
			for (int j=0; j<LOD;j++) {
				float t = float(count2)*float(1)/float(LOD*(NUM_POINTS-1));
				//cout << "t: " << t << endl;
				y = evalLagrange(t, 1, lindex, uindex);
				V.col(count) = y;
				//cout << "index: " << count << endl;
				//cout << "vertex: " << y(0) << ", " << y(1) << endl;
				count++;
				count2++;
			}
		}
		if (seg == 3) {
			V.col(count) = P_sq.col(0);
			//cout << "index: " << count << endl;
			//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
			count++;
		}
	}

	for (int i=1; i<4*NUM_POINTS-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1 || i == 3*NUM_POINTS || i == 3*NUM_POINTS-1)
			continue;
		V.col(count) = P_sq.col(i);
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}

}

void lagrange_hex() {

	V.resize(2,LOD*(NUM_POINTS-1)*6+1+6*NUM_POINTS-12);

	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//cout << "P_hex: " << P_hex.transpose().format(OctaveFmt) << endl;

	int count = 0;
	Vector2f y;
	for (int seg = 0; seg<6; seg++) {
		float lindex = seg*NUM_POINTS;
		float uindex = (seg+1)*NUM_POINTS-1;
		int count2 = 0;
		for (int i=0; i<NUM_POINTS-1;i++) {
			for (int j=0; j<LOD;j++) {
				float t = float(count2)*float(1)/float(LOD*(NUM_POINTS-1));
				//cout << "t: " << t << endl;
				y = evalLagrange(t, 2, lindex, uindex);
				V.col(count) = y;
				//cout << "index: " << count << endl;
				//cout << "vertex: " << y(0) << ", " << y(1) << endl;
				count++;
				count2++;
			}
		}

		if (seg == 5) {
			V.col(count) = P_hex.col(0);
			//cout << "index: " << count << endl;
			//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
			count++;
		}
	}

	for (int i=1; i<6*NUM_POINTS-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1 || i == 3*NUM_POINTS || i == 3*NUM_POINTS-1
			|| i == 4*NUM_POINTS || i == 4*NUM_POINTS-1 || i == 5*NUM_POINTS || i == 5*NUM_POINTS-1)
			continue;
		V.col(count) = P_hex.col(i);
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}
}

void tessellate_sq() {

	V.resize(2,LOD*NUM_POINTS*4+1+4*NUM_POINTS-8);

	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//cout << "P_sq: " << P_sq.transpose().format(OctaveFmt) << endl;

	int count = 0;
	for (int seg = 0; seg<4; seg++) {

		for(int start_cv=-2,j=1;j!=NUM_POINTS+1;++j,++start_cv)
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
				float x = b0 * GetPoint_sq( start_cv + 0 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b1 * GetPoint_sq( start_cv + 1 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b2 * GetPoint_sq( start_cv + 2 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b3 * GetPoint_sq( start_cv + 3 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) ;

				float y = b0 * GetPoint_sq( start_cv + 0 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b1 * GetPoint_sq( start_cv + 1 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b2 * GetPoint_sq( start_cv + 2 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b3 * GetPoint_sq( start_cv + 3 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) ;

				V.col(count) << x,y;
				//cout << "index: " << count << endl;
				//cout << "vertex: " << x << ", " << y << endl;
				count++;
			}
		}
		if (seg == 3) {
			V.col(count) = P_sq.col(0);
			//cout << "index: " << count << endl;
			//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
			count++;
		}
	}

	for (int i=1; i<4*NUM_POINTS-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1 || i == 3*NUM_POINTS || i == 3*NUM_POINTS-1)
			continue;
		V.col(count) = P_sq.col(i);
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}

}

void tessellate_hex() {

	V.resize(2,LOD*NUM_POINTS*6+1+6*NUM_POINTS-12);

	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
	//cout << "P_hex: " << P_hex.transpose().format(OctaveFmt) << endl;

	int count = 0;
	for (int seg = 0; seg<6; seg++) {

		for(int start_cv=-2,j=1;j!=NUM_POINTS+1;++j,++start_cv)
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
				float x = b0 * GetPoint_hex( start_cv + 0 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b1 * GetPoint_hex( start_cv + 1 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b2 * GetPoint_hex( start_cv + 2 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) +
						  b3 * GetPoint_hex( start_cv + 3 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(0) ;

				float y = b0 * GetPoint_hex( start_cv + 0 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b1 * GetPoint_hex( start_cv + 1 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b2 * GetPoint_hex( start_cv + 2 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) +
						  b3 * GetPoint_hex( start_cv + 3 , NUM_POINTS*(seg+1), NUM_POINTS*seg)(1) ;

				V.col(count) << x,y;
				//cout << "index: " << count << endl;
				//cout << "vertex: " << x << ", " << y << endl;
				count++;
			}
		}
		if (seg == 5) {
			V.col(count) = P_hex.col(0);
			//cout << "index: " << count << endl;
			//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
			count++;
		}
	}

	for (int i=1; i<6*NUM_POINTS-1; i++) {
		if (i == NUM_POINTS || i == NUM_POINTS-1 || i == 2*NUM_POINTS || i == 2*NUM_POINTS-1 || i == 3*NUM_POINTS || i == 3*NUM_POINTS-1
			|| i == 4*NUM_POINTS || i == 4*NUM_POINTS-1 || i == 5*NUM_POINTS || i == 5*NUM_POINTS-1)
			continue;
		V.col(count) = P_hex.col(i);
		//cout << "index: " << count << endl;
		//cout << "vertex: " << V(0,count) << ", " << V(1,count) << endl;
		count++;
	}
}

void zoom(float z) {
	if (z < 0.0 && view(0,0) > 0.16 && view(1,1) > 0.16) {
	    view(0,0) *= (z+1);
	    view(1,1) *= (z+1);
	} else if (z > 0.0) {
		view(0,0) *= (z+1);
	    view(1,1) *= (z+1);
	}
}

void rotate(float deg) {
	deg = deg*(PI/180);
	Matrix2f rot;
	rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	view.block(0,0,2,2) = rot*view.block(0,0,2,2);
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
            //cout << "triangle clicked" << endl;
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

int point_sq_click(double xworld, double yworld) {
	float radius = 0.05;
	for (int i=0; i<P_sq.cols(); i++) {
		if (abs(xworld-P_sq(0,i)) < radius && abs(yworld-P_sq(1,i)) < radius) {
			return i;
		}
	}
	return -1;
}

int point_hex_click(double xworld, double yworld) {
	float radius = 0.05;
	for (int i=0; i<P_hex.cols(); i++) {
		if (abs(xworld-P_hex(0,i)) < radius && abs(yworld-P_hex(1,i)) < radius) {
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
	if (pIndex != -1) {
	    // Get the size of the window
	    int width, height;
	    glfwGetWindowSize(window, &width, &height);

	    // Convert screen position to world coordinates
	    Vector4f p_screen(xpos,height-1-ypos,0,1); // NOTE: y axis is flipped in glfw
	    Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1);
	    Vector4f p_world = view.inverse()*p_canonical;

	    double xworld = p_world(0);
	    double yworld = p_world(1);


	    if (trMode) {
	    	int oIndex = trMap.find(pIndex)->second;
	    		if (pIndex != oIndex) {
	    			dx = float(xworld) - xstart;
	    			dy = float(yworld) - ystart;
	    			if (pIndex < NUM_POINTS || oIndex <NUM_POINTS) {
	    				P_tr.col(oIndex) << xstart2+dx, ystart2-dy;
	    			} else {
	    				P_tr.col(oIndex) << xstart2-dx, ystart2-dy;
	    			}
	    		}
	    		P_tr.col(pIndex) << xworld, yworld;
	    		tessellate_triangle();
	    } else if (lagTrMode) {
	    	int oIndex = trMap.find(pIndex)->second;
	    		if (pIndex != oIndex) {
	    			dx = float(xworld) - xstart;
	    			dy = float(yworld) - ystart;
	    			if (pIndex < NUM_POINTS || oIndex <NUM_POINTS) {
	    				P_tr.col(oIndex) << xstart2+dx, ystart2-dy;
	    			} else {
	    				P_tr.col(oIndex) << xstart2-dx, ystart2-dy;
	    			}
	    		}
	    		P_tr.col(pIndex) << xworld, yworld;
	    		lagrange_triangle();
	    } else if (sqMode) {
	    		int oIndex = sqMap.find(pIndex)->second;
	    		P_sq.col(pIndex) << xworld, yworld;
	    		if (pIndex < NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld, yworld+1;
	    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld-1, yworld;
	    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld, yworld-1;
	    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld+1, yworld;
	    		}
	    		tessellate_sq();
	    } else if (lagSqMode) {
		    		int oIndex = sqMap.find(pIndex)->second;
		    		P_sq.col(pIndex) << xworld, yworld;
		    		if (pIndex < NUM_POINTS) {
		    			P_sq.col(oIndex) << xworld, yworld+1;
		    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
		    			P_sq.col(oIndex) << xworld-1, yworld;
		    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
		    			P_sq.col(oIndex) << xworld, yworld-1;
		    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
		    			P_sq.col(oIndex) << xworld+1, yworld;
		    		}
		    		lagrange_sq();
	    } else if (hexMode) {
		    		int oIndex = hexMap.find(pIndex)->second;
		    		Matrix2f rot;
		    		Vector2f centre;
		    		Vector2f tmp;
		    		P_hex.col(pIndex) << xworld, yworld;
		    		tmp << xworld, yworld;
		    		if (pIndex < NUM_POINTS) {
		    			float deg = -120*(PI/180);
		    			centre = P_hex.col(NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
		    			float deg = 120*(PI/180);
		    			centre = P_hex.col(NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
		    			float deg = -120*(PI/180);
		    			centre = P_hex.col(3*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
		    			float deg = 120*(PI/180);
		    			centre = P_hex.col(3*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 4*NUM_POINTS && pIndex < 5*NUM_POINTS) {
		    			float deg = -120*(PI/180);
		    			centre = P_hex.col(5*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 5*NUM_POINTS && pIndex < 6*NUM_POINTS) {
		    			float deg = 120*(PI/180);
		    			centre = P_hex.col(5*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		}
		    		tessellate_hex();
		} else if (lagHexMode) {
		    		int oIndex = hexMap.find(pIndex)->second;
		    		Matrix2f rot;
		    		Vector2f centre;
		    		Vector2f tmp;
		    		P_hex.col(pIndex) << xworld, yworld;
		    		tmp << xworld, yworld;
		    		if (pIndex < NUM_POINTS) {
		    			float deg = -120*(PI/180);
		    			centre = P_hex.col(NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
		    			float deg = 120*(PI/180);
		    			centre = P_hex.col(NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
		    			float deg = -120*(PI/180);
		    			centre = P_hex.col(3*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
		    			float deg = 120*(PI/180);
		    			centre = P_hex.col(3*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 4*NUM_POINTS && pIndex < 5*NUM_POINTS) {
		    			float deg = -120*(PI/180);
		    			centre = P_hex.col(5*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		} else if (pIndex > 5*NUM_POINTS && pIndex < 6*NUM_POINTS) {
		    			float deg = 120*(PI/180);
		    			centre = P_hex.col(5*NUM_POINTS);
		    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
		    			tmp = tmp-centre;
		    			tmp = rot*tmp;
		    			tmp = tmp+centre;
		    			P_hex.col(oIndex) = tmp;
		    		}
		    		lagrange_hex();
		 }

		VBO.update(V);
	}
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
	    		xstart = float(xworld);
	    		ystart = float(yworld);
	    		int oIndex = trMap.find(pIndex)->second;
	    		if (pIndex != oIndex) { 	
	    			xstart2 = P_tr(0,oIndex);
	    			ystart2 = P_tr(1,oIndex);
	    		}
	    		//cout << "index clicked: " << pIndex << endl;
	    		//cout << "vertex: " << P_tr(0,pIndex) << " , " << P_tr(1,pIndex) << endl;
	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		P_tr.col(pIndex) << xworld, yworld;
	    		int oIndex = trMap.find(pIndex)->second;
	    		if (pIndex != oIndex) {
	    			if (pIndex < NUM_POINTS || oIndex <NUM_POINTS) {
	    				P_tr.col(oIndex) << xstart2+dx, ystart2-dy;
	    			} else {
	    				P_tr.col(oIndex) << xstart2-dx, ystart2-dy;
	    			}
	    		}
	    		tessellate_triangle();
	    		pIndex = -1;
	    	}
	    }
    } else if (lagTrMode) {
    	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
	    	pIndex = point_tr_click(xworld,yworld);
	    	if (pIndex != -1) {
	    		xstart = float(xworld);
	    		ystart = float(yworld);
	    		int oIndex = trMap.find(pIndex)->second;
	    		if (pIndex != oIndex) { 	
	    			xstart2 = P_tr(0,oIndex);
	    			ystart2 = P_tr(1,oIndex);
	    		}
	    		//cout << "index clicked: " << pIndex << endl;
	    		//cout << "vertex: " << P_tr(0,pIndex) << " , " << P_tr(1,pIndex) << endl;
	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		P_tr.col(pIndex) << xworld, yworld;
	    		int oIndex = trMap.find(pIndex)->second;
	    		if (pIndex != oIndex) {
	    			if (pIndex < NUM_POINTS || oIndex <NUM_POINTS) {
	    				P_tr.col(oIndex) << xstart2+dx, ystart2-dy;
	    			} else {
	    				P_tr.col(oIndex) << xstart2-dx, ystart2-dy;
	    			}
	    		}
	    		lagrange_triangle();
	    		pIndex = -1;
	    	}
	    }
    } else if (sqMode) {
    	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
	    	pIndex = point_sq_click(xworld,yworld);
	    	if (pIndex != -1) {
	    		//cout << "index clicked: " << pIndex << endl;
	    		//cout << "vertex: " << P_sq(0,pIndex) << " , " << P_sq(1,pIndex) << endl;

	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		int oIndex = sqMap.find(pIndex)->second;
	    		P_sq.col(pIndex) << xworld, yworld;
	    		if (pIndex < NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld, yworld+1;
	    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld-1, yworld;
	    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld, yworld-1;
	    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld+1, yworld;
	    		}
	    		tessellate_sq();
	    		pIndex = -1;
	    	}
	    }
    } else if (lagSqMode) {
    	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
	    	pIndex = point_sq_click(xworld,yworld);
	    	if (pIndex != -1) {
	    		//cout << "index clicked: " << pIndex << endl;
	    		//cout << "vertex: " << P_sq(0,pIndex) << " , " << P_sq(1,pIndex) << endl;

	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		int oIndex = sqMap.find(pIndex)->second;
	    		P_sq.col(pIndex) << xworld, yworld;
	    		if (pIndex < NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld, yworld+1;
	    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld-1, yworld;
	    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld, yworld-1;
	    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
	    			P_sq.col(oIndex) << xworld+1, yworld;
	    		}
	    		lagrange_sq();
	    		pIndex = -1;
	    	}
	    }
    } else if (hexMode) {
    	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
	    	pIndex = point_hex_click(xworld,yworld);
	    	if (pIndex != -1) {
	    		//cout << "index clicked: " << pIndex << endl;
	    		//cout << "vertex: " << P_hex(0,pIndex) << " , " << P_hex(1,pIndex) << endl;
	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		int oIndex = hexMap.find(pIndex)->second;
	    		Matrix2f rot;
	    		Vector2f centre;
	    		Vector2f tmp;
	    		P_hex.col(pIndex) << xworld, yworld;
	    		tmp << xworld, yworld;
	    		if (pIndex < NUM_POINTS) {
	    			float deg = -120*(PI/180);
	    			centre = P_hex.col(NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
	    			float deg = 120*(PI/180);
	    			centre = P_hex.col(NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
	    			float deg = -120*(PI/180);
	    			centre = P_hex.col(3*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
	    			float deg = 120*(PI/180);
	    			centre = P_hex.col(3*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 4*NUM_POINTS && pIndex < 5*NUM_POINTS) {
	    			float deg = -120*(PI/180);
	    			centre = P_hex.col(5*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 5*NUM_POINTS && pIndex < 6*NUM_POINTS) {
	    			float deg = 120*(PI/180);
	    			centre = P_hex.col(5*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		}
	    		tessellate_hex();
	    		pIndex = -1;
	    	}
	    }
	} else if (lagHexMode) {
    	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
	    	pIndex = point_hex_click(xworld,yworld);
	    	if (pIndex != -1) {
	    		//cout << "index clicked: " << pIndex << endl;
	    		//cout << "vertex: " << P_hex(0,pIndex) << " , " << P_hex(1,pIndex) << endl;
	    	}
	    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
	    	if (pIndex != -1) {
	    		int oIndex = hexMap.find(pIndex)->second;
	    		Matrix2f rot;
	    		Vector2f centre;
	    		Vector2f tmp;
	    		P_hex.col(pIndex) << xworld, yworld;
	    		tmp << xworld, yworld;
	    		if (pIndex < NUM_POINTS) {
	    			float deg = -120*(PI/180);
	    			centre = P_hex.col(NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > NUM_POINTS && pIndex < 2*NUM_POINTS) {
	    			float deg = 120*(PI/180);
	    			centre = P_hex.col(NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 2*NUM_POINTS && pIndex < 3*NUM_POINTS) {
	    			float deg = -120*(PI/180);
	    			centre = P_hex.col(3*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 3*NUM_POINTS && pIndex < 4*NUM_POINTS) {
	    			float deg = 120*(PI/180);
	    			centre = P_hex.col(3*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 4*NUM_POINTS && pIndex < 5*NUM_POINTS) {
	    			float deg = -120*(PI/180);
	    			centre = P_hex.col(5*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		} else if (pIndex > 5*NUM_POINTS && pIndex < 6*NUM_POINTS) {
	    			float deg = 120*(PI/180);
	    			centre = P_hex.col(5*NUM_POINTS);
	    			rot << cos(deg), -sin(deg), sin(deg), cos(deg);
	    			tmp = tmp-centre;
	    			tmp = rot*tmp;
	    			tmp = tmp+centre;
	    			P_hex.col(oIndex) = tmp;
	    		}
	    		lagrange_hex();
	    		pIndex = -1;
	    	}
	   	}
	 }

    // Upload the change to the GPU
    VBO.update(V);
}

void colorV() {

	fillEdges();
	MatrixXf Vin = prepareVin();
	MatrixXf Vout;
	MatrixXi Eout;
	MatrixXf H;
  	//H.resize(1,2);
  	//H << 0,0;
	//Vin.resize(8,2);
  	//Ein.resize(8,2);
	//Vin << -1,-1, 1,-1, 1,1, -1, 1,
       //-2,-2, 2,-2, 2,2, -2, 2;
  	//Ein << 0,1, 1,2, 2,3, 3,0,
       //4,5, 5,6, 6,7, 7,4;

  	//IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
  	//cout << "Vin rows: " << Vin.rows() << endl;
	//cout << "Vin: " << Vin.format(OctaveFmt) << endl;
	//cout << "Ein: " << E.format(OctaveFmt) << endl;

	mytriangulate(Vin,E,H,"",Vout,Eout);

	//cout << "V cols: " << V.cols() << endl;
	//cout << "Vout: " << Vout.format(OctaveFmt) << endl;
	//cout << "Eout: " << Eout.format(OctaveFmt) << endl;

	int tmpCols = V.cols();
	prevCols = V.cols();
	numCols = 3*Eout.rows();
	V.conservativeResize(2,tmpCols+3*Eout.rows());
	for (int i=0; i<Eout.rows(); i++) {
		V.col(tmpCols) = Vout.row(Eout(i,0));
		V.col(tmpCols+1) = Vout.row(Eout(i,1));
		V.col(tmpCols+2) = Vout.row(Eout(i,2));
		tmpCols += 3;
	}

	//cout << "Vfinal: " << V.transpose().format(OctaveFmt) << endl;

}

void window_size_callback(GLFWwindow* window, int width, int height)
{
	float aspect_ratio = float(height)/float(width); // corresponds to the necessary width scaling
    view <<
    aspect_ratio,0, 0, 0,
    0,           1, 0, 0,
    0,           0, 1, 0,
    0,           0, 0, 1;

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_RELEASE) {
	    switch (key)
	    {
            case GLFW_KEY_T:
            	setMode(true,false,false,false,false,false,false,false,false,false,false,false);
            	tessellate_triangle();
            	break;
            case GLFW_KEY_1:
            	setMode(false,true,false,false,false,false,false,false,false,false,false,false);
            	lagrange_triangle();
            	break;
            case GLFW_KEY_S:
            	setMode(false,false,false,false,true,false,false,false,false,false,false,false);
            	tessellate_sq();
            	break;
            case GLFW_KEY_2:
            	setMode(false,false,false,false,false,true,false,false,false,false,false,false);
            	lagrange_sq();
            	break;
            case GLFW_KEY_H:
            	setMode(false,false,false,false,false,false,false,false,true,false,false,false);
            	tessellate_hex();
            	break;
            case GLFW_KEY_3:
             	setMode(false,false,false,false,false,false,false,false,false,true,false,false);
            	lagrange_hex();
            	break;
            case GLFW_KEY_X:
            	if (trMode) {
            		setMode(false,false,true,false,false,false,false,false,false,false,false,false);
            		colorV();
            	} else if (sqMode) {
            		setMode(false,false,false,false,false,false,true,false,false,false,false,false);
            		colorV();
            	} else if (hexMode) {
            		setMode(false,false,false,false,false,false,false,false,false,false,true,false);
            		colorV();
            	} else if (lagTrMode) {
            		setMode(false,false,false,true,false,false,false,false,false,false,false,false);
            		colorV();
            	} else if (lagSqMode) {
            		setMode(false,false,false,false,false,false,false,true,false,false,false,false);
            		colorV();
            	} else if (lagHexMode) {
            		setMode(false,false,false,false,false,false,false,false,false,false,false,true);
            		colorV();
            	}
            	break;
            case GLFW_KEY_EQUAL:
            	zoom(0.05);
            	break;
            case GLFW_KEY_MINUS:
            	zoom(-0.05);
            	break;
            case GLFW_KEY_Q:
            	rotate(5);
            	break;
            case GLFW_KEY_W:
            	rotate(-5);
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

    //control points for triangle (change)
	float rad = 0.5;
	float theta = 90*(PI/180);
	MatrixXf ctrl_tr(3,2);
	ctrl_tr << 0, rad, rad*cos(theta+2*PI/3), rad*sin(theta+2*PI/3), 
	rad*cos(theta+4*PI/3), rad*sin(theta+4*PI/3);
	tr_len = sqrt(3)*rad;
    for (int i=0; i<NUM_POINTS; i++) {
    	P_tr.col(i) << ctrl_tr(0,0)+float(i*(ctrl_tr(1,0)-ctrl_tr(0,0)))/(NUM_POINTS-1),
		ctrl_tr(0,1)+float(i*(ctrl_tr(1,1)-ctrl_tr(0,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_tr.col(i+NUM_POINTS) << ctrl_tr(1,0)+float(i*(ctrl_tr(2,0)-ctrl_tr(1,0)))/(NUM_POINTS-1),
		ctrl_tr(1,1)+float(i*(ctrl_tr(2,1)-ctrl_tr(1,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_tr.col(i+2*NUM_POINTS) << ctrl_tr(2,0)+float(i*(ctrl_tr(0,0)-ctrl_tr(2,0)))/(NUM_POINTS-1),
		ctrl_tr(2,1)+float(i*(ctrl_tr(0,1)-ctrl_tr(2,1)))/(NUM_POINTS-1);
	}

	// triangle coord for pointy tess
	Tr << 0, rad, rad*cos(theta+2*PI/3), rad*sin(theta+2*PI/3), 
	0, rad*sin(theta+2*PI/3), rad*cos(theta+4*PI/3), rad*sin(theta+4*PI/3);

	//control points for square
	// -0.5, 0.5, 0.5, -.5
	// -0.5, -.5, 0.5, 0.5
	for (int i=0; i<NUM_POINTS; i++) {
		P_sq.col(i) << -0.5+float(i*1)/(NUM_POINTS-1),-0.5;
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_sq.col(i+NUM_POINTS) << 0.5, -0.5+float(i*1)/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_sq.col(i+2*NUM_POINTS) << 0.5-float(i*1)/(NUM_POINTS-1),0.5;
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_sq.col(i+3*NUM_POINTS) << -0.5,0.5-float(i*1)/(NUM_POINTS-1);
	}

	//sqMap --change with num_points
	trMap.insert(pair<int, int>(1, 11));
	trMap.insert(pair<int, int>(11, 1));
	trMap.insert(pair<int, int>(2, 12));
	trMap.insert(pair<int, int>(12, 2));
	trMap.insert(pair<int, int>(3, 13));
	trMap.insert(pair<int, int>(13, 3));
	trMap.insert(pair<int, int>(6, 8));
	trMap.insert(pair<int, int>(8, 6));
	trMap.insert(pair<int, int>(7, 7));

	//sqMap --change with num_points
	sqMap.insert(pair<int, int>(1, 13));
	sqMap.insert(pair<int, int>(2, 12));
	sqMap.insert(pair<int, int>(3, 11));
	sqMap.insert(pair<int, int>(6, 18));
	sqMap.insert(pair<int, int>(7, 17));
	sqMap.insert(pair<int, int>(8, 16));
	sqMap.insert(pair<int, int>(13, 1));
	sqMap.insert(pair<int, int>(12, 2));
	sqMap.insert(pair<int, int>(11, 3));
	sqMap.insert(pair<int, int>(18, 6));
	sqMap.insert(pair<int, int>(17, 7));
	sqMap.insert(pair<int, int>(16, 8));

	//hexMap --change with num_points
	hexMap.insert(pair<int, int>(2, 7));
	hexMap.insert(pair<int, int>(7, 2));
	hexMap.insert(pair<int, int>(1, 8));
	hexMap.insert(pair<int, int>(8, 1));
	hexMap.insert(pair<int, int>(3, 6));
	hexMap.insert(pair<int, int>(6, 3));
	hexMap.insert(pair<int, int>(11, 18));
	hexMap.insert(pair<int, int>(18, 11));
	hexMap.insert(pair<int, int>(12, 17));
	hexMap.insert(pair<int, int>(17, 12));
	hexMap.insert(pair<int, int>(13, 16));
	hexMap.insert(pair<int, int>(16, 13));
	hexMap.insert(pair<int, int>(21, 28));
	hexMap.insert(pair<int, int>(28, 21));
	hexMap.insert(pair<int, int>(22, 27));
	hexMap.insert(pair<int, int>(27, 22));
	hexMap.insert(pair<int, int>(23, 26));
	hexMap.insert(pair<int, int>(26, 23));

	//control points for hexagon
	rad = 0.5;
	MatrixXf ctrl_hex(6,2);
	ctrl_hex << rad, 0, rad/2, sqrt(3)*rad/2, -rad/2, sqrt(3)*rad/2, 
	-rad, 0, -rad/2, -sqrt(3)*rad/2, rad/2, -sqrt(3)*rad/2;
	for (int i=0; i<NUM_POINTS; i++) {
		P_hex.col(i) << ctrl_hex(0,0)+float(i*(ctrl_hex(1,0)-ctrl_hex(0,0)))/(NUM_POINTS-1),
		ctrl_hex(0,1)+float(i*(ctrl_hex(1,1)-ctrl_hex(0,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_hex.col(i+NUM_POINTS) << ctrl_hex(1,0)+float(i*(ctrl_hex(2,0)-ctrl_hex(1,0)))/(NUM_POINTS-1),
		ctrl_hex(1,1)+float(i*(ctrl_hex(2,1)-ctrl_hex(1,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_hex.col(i+2*NUM_POINTS) << ctrl_hex(2,0)+float(i*(ctrl_hex(3,0)-ctrl_hex(2,0)))/(NUM_POINTS-1),
		ctrl_hex(2,1)+float(i*(ctrl_hex(3,1)-ctrl_hex(2,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_hex.col(i+3*NUM_POINTS) << ctrl_hex(3,0)+float(i*(ctrl_hex(4,0)-ctrl_hex(3,0)))/(NUM_POINTS-1),
		ctrl_hex(3,1)+float(i*(ctrl_hex(4,1)-ctrl_hex(3,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_hex.col(i+4*NUM_POINTS) << ctrl_hex(4,0)+float(i*(ctrl_hex(5,0)-ctrl_hex(4,0)))/(NUM_POINTS-1),
		ctrl_hex(4,1)+float(i*(ctrl_hex(5,1)-ctrl_hex(4,1)))/(NUM_POINTS-1);
	}
	for (int i=0; i<NUM_POINTS; i++) {
		P_hex.col(i+5*NUM_POINTS) << ctrl_hex(5,0)+float(i*(ctrl_hex(0,0)-ctrl_hex(5,0)))/(NUM_POINTS-1),
		ctrl_hex(5,1)+float(i*(ctrl_hex(0,1)-ctrl_hex(5,1)))/(NUM_POINTS-1);
	}

	t_ar << 0, 0.25, 0.5, 0.75, 1;
    view.setIdentity();
    model.setIdentity();

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    Program program;
    const GLchar* vertex_shader =
            "#version 150 core\n"
                    "in vec2 position;"
                    "uniform mat4 view;"
                    "uniform mat4 model;"
                    "void main()"
                    "{"
                    "    gl_Position = view*model*vec4(position,0.0,1.0);"
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

    glfwSetWindowSizeCallback(window, window_size_callback);

    // cursor movement callback
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    

    // Loop until the user closes the window
    float start;
    float starty;
    float offset;

    while (!glfwWindowShouldClose(window))
    {
        // Bind your VAO (not necessary if you have only one)
        VAO.bind();

        // Bind your program
        program.bind();

        model.setIdentity();
        glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());
        glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());

        // Clear the framebuffer
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

		if (trMode) {

			glPointSize(5);
			glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.0);
			//glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+1, 3*NUM_POINTS-6);

			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+1, 1);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+1+6, 1); // change with num points

			glUniform3f(program.uniform("triangleColor"),0.8,0.0,0.9);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+2, 1);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+2+6, 1);

			glUniform3f(program.uniform("triangleColor"),0.6,0.2,0.0);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+3, 1);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+3+6, 1);

			glUniform3f(program.uniform("triangleColor"),0.2,0.0,0.6);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+4, 1);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+4+2, 1);

			glUniform3f(program.uniform("triangleColor"),1.0,1.0,1.0);
			glDrawArrays(GL_POINTS, 3*NUM_POINTS*LOD+5, 1);

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			glDrawArrays(GL_LINE_STRIP, 0, 3*NUM_POINTS*LOD+1);

		} else if (finalMode) {
			
			//horizontal
			for (int i=0; i<19; i++) {
				model(0,3) = tr_len*(9-i);
				model(1,3) = 0;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 3*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			starty = sqrt(3)*0.5*tr_len+5*sqrt(3)*tr_len;
			for (int j=0; j<10; j++) {
				//up/down
				start = tr_len*0.5*17;
				for (int i=0; i<19; i++) {
					model(0,3) = start-i*tr_len;
					model(1,3) = starty-j*sqrt(3)*tr_len;
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 3*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

			for (int j=0; j<5; j++) {
				//upup
				for (int i=0; i<19; i++) {
					model(0,3) = tr_len*(9-i);
					model(1,3) = sqrt(3)*tr_len*(j+1);
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 3*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
				//downdown
				for (int i=0; i<19; i++) {
					model(0,3) = tr_len*(9-i);
					model(1,3) = -sqrt(3)*tr_len*(j+1);
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 3*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

		} else if (lagTrMode) {

			glPointSize(5);
			glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.0);
			//glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+1, 3*NUM_POINTS-6);

			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+1, 1);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+1+6, 1); // change with num points

			glUniform3f(program.uniform("triangleColor"),0.8,0.0,0.9);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+2, 1);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+2+6, 1);

			glUniform3f(program.uniform("triangleColor"),0.6,0.2,0.0);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+3, 1);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+3+6, 1);

			glUniform3f(program.uniform("triangleColor"),0.2,0.0,0.6);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+4, 1);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+4+2, 1);

			glUniform3f(program.uniform("triangleColor"),1.0,1.0,1.0);
			glDrawArrays(GL_POINTS, 3*(NUM_POINTS-1)*LOD+5, 1);

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			glDrawArrays(GL_LINE_STRIP, 0, 3*(NUM_POINTS-1)*LOD+1);

		} else if (finalLagMode) {
			
			//horizontal
			for (int i=0; i<19; i++) {
				model(0,3) = tr_len*(9-i);
				model(1,3) = 0;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 3*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			starty = sqrt(3)*0.5*tr_len+5*sqrt(3)*tr_len;
			for (int j=0; j<10; j++) {
				//up/down
				start = tr_len*0.5*17;
				for (int i=0; i<19; i++) {
					model(0,3) = start-i*tr_len;
					model(1,3) = starty-j*sqrt(3)*tr_len;
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 3*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

			for (int j=0; j<5; j++) {
				//upup
				for (int i=0; i<19; i++) {
					model(0,3) = tr_len*(9-i);
					model(1,3) = sqrt(3)*tr_len*(j+1);
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 3*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
				//downdown
				for (int i=0; i<19; i++) {
					model(0,3) = tr_len*(9-i);
					model(1,3) = -sqrt(3)*tr_len*(j+1);
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 3*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.5,0.1,0.1);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

		} else if (sqMode) {
			glPointSize(5);
			int count = 0;
			for (int i=4*NUM_POINTS*LOD+1; i<(4*NUM_POINTS*LOD+1+4*NUM_POINTS-8); i+=(NUM_POINTS-2)) {
				if (count%2 == 0) {
					glUniform3f(program.uniform("triangleColor"),1.0,0.1,0.6);
				} else {
					glUniform3f(program.uniform("triangleColor"),1.0,0.8,0.3);
				}
				glDrawArrays(GL_POINTS, i, (NUM_POINTS-2));
				count++;
			}
			//glDrawArrays(GL_POINTS, 4*NUM_POINTS*LOD+1, 4*NUM_POINTS-8);
			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			glDrawArrays(GL_LINE_STRIP, 0, 4*NUM_POINTS*LOD+1);
		} else if (finalsqMode) {

			//horizontal
			for (int i=0; i<19; i++) {
				model(0,3) = 16-2*i;
				model(1,3) = 0;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 4*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),0.7,0.0,0.0);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			for (int j=0; j<9; j++) {		
				//up/down
				for (int i=0; i<18; i++) {
					model(0,3) = 15-2*i;
					model(1,3) = 7-2*j;
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 4*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.7,0.0,0.0);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

			for (int j=0; j<10; j++) { 
				//upup/downdown
				for (int i=0; i<19; i++) {
					model(0,3) = 16-2*i;
					model(1,3) = 8-2*j;
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 4*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.7,0.0,0.0);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

		} else if (lagSqMode) {
			glPointSize(5);
			//glDrawArrays(GL_POINTS, 4*(NUM_POINTS-1)*LOD+1, 4*NUM_POINTS-8);
			int count = 0;
			for (int i=4*(NUM_POINTS-1)*LOD+1; i<(4*(NUM_POINTS-1)*LOD+1+4*NUM_POINTS-8); i+=(NUM_POINTS-2)) {
				if (count%2 == 0) {
					glUniform3f(program.uniform("triangleColor"),1.0,0.1,0.6);
				} else {
					glUniform3f(program.uniform("triangleColor"),1.0,0.8,0.3);
				}
				glDrawArrays(GL_POINTS, i, (NUM_POINTS-2));
				count++;
			}

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			glDrawArrays(GL_LINE_STRIP, 0, 4*(NUM_POINTS-1)*LOD+1);
		} else if (finalLagSqMode) {

			//horizontal
			for (int i=0; i<19; i++) {
				model(0,3) = 16-2*i;
				model(1,3) = 0;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 4*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),0.7,0.0,0.0);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			for (int j=0; j<9; j++) {		
				//up/down
				for (int i=0; i<18; i++) {
					model(0,3) = 15-2*i;
					model(1,3) = 7-2*j;
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 4*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.7,0.0,0.0);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}

			}

			for (int j=0; j<10; j++) { 
				//upup/downdown
				for (int i=0; i<19; i++) {
					model(0,3) = 16-2*i;
					model(1,3) = 8-2*j;
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 4*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),0.7,0.0,0.0);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

		} else if (hexMode) {
			glPointSize(5);
			//glDrawArrays(GL_POINTS, 6*NUM_POINTS*LOD+1, 6*NUM_POINTS-12);
			int count = 0;
			for (int i=6*NUM_POINTS*LOD+1; i<(6*NUM_POINTS*LOD+1+6*NUM_POINTS-12); i+=(NUM_POINTS-2)) {
				if (count < 2) {
					glUniform3f(program.uniform("triangleColor"),0.4,0.1,0.2);
				} else if (count < 4) {
					glUniform3f(program.uniform("triangleColor"),0.0,0.1,0.6);
				} else {
					glUniform3f(program.uniform("triangleColor"),0.0,0.4,0.2);
				}
				glDrawArrays(GL_POINTS, i, (NUM_POINTS-2));
				count++;
			}

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
		} else if (finalhexMode) {

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);

			//horizontal
			for (int i=0; i<19; i++) {

				float deg1 = 120*(PI/180);
				model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = 0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = 0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = -sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				float deg2 = -120*(PI/180);
				model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = -0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = -0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			//up
			starty = 0.75*sqrt(3);
			offset = 0.75;
			for (int j=0; j<2; j++) {
				for (int i=0; i<19; i++) {

					float deg1 = 120*(PI/180);
					model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty+(j*3*0.5*sqrt(3))+0.25*sqrt(3);
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty+0.25*sqrt(3)+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty-sqrt(3)*0.5+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					float deg2 = -120*(PI/180);
					model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty+sqrt(3)*0.5+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}
			//down
			starty = -0.75*sqrt(3);
			offset = 0.75;
			for (int j=0; j<2; j++) {
				for (int i=0; i<19; i++) {

					float deg1 = 120*(PI/180);
					model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty+0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty+0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty-sqrt(3)*0.5-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					float deg2 = -120*(PI/180);
					model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty+sqrt(3)*0.5-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

			//upup
			starty = 3*0.5*sqrt(3);
			for (int i=0; i<19; i++) {

				float deg1 = 120*(PI/180);
				model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty-sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				float deg2 = -120*(PI/180);
				model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty+sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			//downdown
			starty = -3*0.5*sqrt(3);
			for (int i=0; i<19; i++) {

				float deg1 = 120*(PI/180);
				model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty-sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				float deg2 = -120*(PI/180);
				model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty+sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*NUM_POINTS*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

		} else if (lagHexMode) {
			glPointSize(5);
			//glDrawArrays(GL_POINTS, 6*(NUM_POINTS-1)*LOD+1, 6*NUM_POINTS-12);
			int count = 0;
			for (int i=6*(NUM_POINTS-1)*LOD+1; i<(6*(NUM_POINTS-1)*LOD+1+6*NUM_POINTS-12); i+=(NUM_POINTS-2)) {
				if (count < 2) {
					glUniform3f(program.uniform("triangleColor"),0.4,0.1,0.2);
				} else if (count < 4) {
					glUniform3f(program.uniform("triangleColor"),0.0,0.1,0.6);
				} else {
					glUniform3f(program.uniform("triangleColor"),0.0,0.4,0.2);
				}
				glDrawArrays(GL_POINTS, i, (NUM_POINTS-2));
				count++;
			}
			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
			glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
		} else if (finalLagHexMode) {

			glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);

			//horizontal
			for (int i=0; i<19; i++) {

				float deg1 = 120*(PI/180);
				model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = 0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = 0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = -sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				float deg2 = -120*(PI/180);
				model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = -0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = -0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			//up
			starty = 0.75*sqrt(3);
			offset = 0.75;
			for (int j=0; j<2; j++) {
				for (int i=0; i<19; i++) {

					float deg1 = 120*(PI/180);
					model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty+(j*3*0.5*sqrt(3))+0.25*sqrt(3);
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty+0.25*sqrt(3)+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty-sqrt(3)*0.5+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					float deg2 = -120*(PI/180);
					model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty+sqrt(3)*0.5+(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}
			//down
			starty = -0.75*sqrt(3);
			offset = 0.75;
			for (int j=0; j<2; j++) {
				for (int i=0; i<19; i++) {

					float deg1 = 120*(PI/180);
					model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty+0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty+0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty-sqrt(3)*0.5-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					float deg2 = -120*(PI/180);
					model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
					model(0,3) = offset-0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0.75+1.5*9-1.5*i;
					model(1,3) = starty-0.25*sqrt(3)-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);

					model(0,3) = offset+0+1.5*9-1.5*i;
					model(1,3) = starty+sqrt(3)*0.5-(j*3*0.5*sqrt(3));
					glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
					glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
					glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
					glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
					glDrawArrays(GL_TRIANGLES, prevCols, numCols);
				}
			}

			//upup
			starty = 3*0.5*sqrt(3);
			for (int i=0; i<19; i++) {

				float deg1 = 120*(PI/180);
				model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty-sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				float deg2 = -120*(PI/180);
				model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty+sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

			//downdown
			starty = -3*0.5*sqrt(3);
			for (int i=0; i<19; i++) {

				float deg1 = 120*(PI/180);
				model.block(0,0,2,2) << cos(deg1), -sin(deg1), sin(deg1), cos(deg1);
				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty+0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty-sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,1.0,0.2);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				float deg2 = -120*(PI/180);
				model.block(0,0,2,2) << cos(deg2), -sin(deg2), sin(deg2), cos(deg2);
				model(0,3) = -0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0.75+1.5*9-1.5*i;
				model(1,3) = starty-0.25*sqrt(3);
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);

				model(0,3) = 0+1.5*9-1.5*i;
				model(1,3) = starty+sqrt(3)*0.5;
				glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, model.data());
				glUniform3f(program.uniform("triangleColor"),0.0,0.0,0.0);
				glDrawArrays(GL_LINE_STRIP, 0, 6*(NUM_POINTS-1)*LOD+1);
				glUniform3f(program.uniform("triangleColor"),1.0,0.0,0.3);
				glDrawArrays(GL_TRIANGLES, prevCols, numCols);
			}

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
